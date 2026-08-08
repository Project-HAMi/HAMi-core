/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "../src/include/hostpid_broker.h"

#define REQUEST_SIZE 8
#define RESPONSE_SIZE 12

typedef void (*server_action_t)(int connection);

static void read_request(int connection) {
    unsigned char request[REQUEST_SIZE];
    size_t offset = 0;

    while (offset < sizeof(request)) {
        ssize_t received = read(connection, request + offset,
                                sizeof(request) - offset);
        assert(received > 0);
        offset += (size_t)received;
    }
    assert(memcmp(request, "HPID\0\1\0\1", sizeof(request)) == 0);
}

static void write_all(int connection, const unsigned char *buffer,
                      size_t length) {
    while (length > 0) {
        ssize_t written = write(connection, buffer, length);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return;
        }
        buffer += written;
        length -= (size_t)written;
    }
}

static void make_response(unsigned char response[RESPONSE_SIZE],
                          uint16_t version, uint16_t status, uint32_t pid) {
    const unsigned char value[RESPONSE_SIZE] = {
        'H', 'P', 'I', 'D',
        (unsigned char)(version >> 8), (unsigned char)version,
        (unsigned char)(status >> 8), (unsigned char)status,
        (unsigned char)(pid >> 24), (unsigned char)(pid >> 16),
        (unsigned char)(pid >> 8), (unsigned char)pid,
    };
    memcpy(response, value, sizeof(value));
}

static void serve_success(int connection) {
    unsigned char response[RESPONSE_SIZE];

    read_request(connection);
    make_response(response, 1, 0, 43210);
    write_all(connection, response, sizeof(response));
}

static void serve_bad_magic(int connection) {
    unsigned char response[RESPONSE_SIZE];

    read_request(connection);
    make_response(response, 1, 0, 43210);
    response[0] = 'B';
    write_all(connection, response, sizeof(response));
}

static void serve_bad_version(int connection) {
    unsigned char response[RESPONSE_SIZE];

    read_request(connection);
    make_response(response, 2, 0, 43210);
    write_all(connection, response, sizeof(response));
}

static void serve_error_status(int connection) {
    unsigned char response[RESPONSE_SIZE];

    read_request(connection);
    make_response(response, 1, 1, 0);
    write_all(connection, response, sizeof(response));
}

static void serve_zero_pid(int connection) {
    unsigned char response[RESPONSE_SIZE];

    read_request(connection);
    make_response(response, 1, 0, 0);
    write_all(connection, response, sizeof(response));
}

static void serve_large_pid(int connection) {
    unsigned char response[RESPONSE_SIZE];

    read_request(connection);
    make_response(response, 1, 0, (uint32_t)INT_MAX + 1U);
    write_all(connection, response, sizeof(response));
}

static void serve_partial_response(int connection) {
    unsigned char response[RESPONSE_SIZE];

    read_request(connection);
    make_response(response, 1, 0, 43210);
    write_all(connection, response, 5);
}

static void serve_trickle_response(int connection) {
    unsigned char response[RESPONSE_SIZE];
    struct timespec delay = {
        .tv_sec = 0,
        .tv_nsec = 30000000L,
    };
    size_t i;

    read_request(connection);
    make_response(response, 1, 0, 43210);
    for (i = 0; i < sizeof(response); i++) {
        if (write(connection, response + i, 1) != 1) {
            return;
        }
        nanosleep(&delay, NULL);
    }
}

static int make_listener(char *socket_path, size_t socket_path_size,
                         char *directory, size_t directory_size) {
    struct sockaddr_un address;
    socklen_t address_length;
    size_t socket_path_length;
    int listener;

    assert(directory_size >= sizeof("/tmp/hami-hostpid-test-XXXXXX"));
    strcpy(directory, "/tmp/hami-hostpid-test-XXXXXX");
    assert(mkdtemp(directory) != NULL);
    assert(snprintf(socket_path, socket_path_size, "%s/broker.sock",
                    directory) > 0);

    listener = socket(AF_UNIX, SOCK_STREAM, 0);
    assert(listener >= 0);
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    socket_path_length = strlen(socket_path);
    if (socket_path_length >= sizeof(address.sun_path)) {
        fprintf(stderr, "host PID broker test socket path is too long\n");
        abort();
    }
    memcpy(address.sun_path, socket_path, socket_path_length + 1);
    address_length =
        (socklen_t)(offsetof(struct sockaddr_un, sun_path) +
                    socket_path_length + 1);
#ifdef __APPLE__
    address.sun_len = address_length;
#endif
    if (bind(listener, (struct sockaddr *)&address, address_length) != 0) {
        perror("bind host PID broker test socket");
        abort();
    }
    assert(listen(listener, 4) == 0);
    return listener;
}

static int64_t run_query_case(server_action_t action, int expected_result,
                              int expected_errno, pid_t expected_pid) {
    char directory[PATH_MAX];
    char socket_path[PATH_MAX];
    pid_t child;
    pid_t host_pid = 99;
    int status;
    struct timespec query_begin;
    struct timespec query_end;
    int listener = make_listener(socket_path, sizeof(socket_path),
                                 directory, sizeof(directory));

    child = fork();
    assert(child >= 0);
    if (child == 0) {
        int connection = accept(listener, NULL, NULL);
        assert(connection >= 0);
        action(connection);
        close(connection);
        close(listener);
        _exit(0);
    }

    errno = 0;
    assert(clock_gettime(CLOCK_MONOTONIC, &query_begin) == 0);
    assert(hostpid_broker_query(socket_path, &host_pid) == expected_result);
    assert(clock_gettime(CLOCK_MONOTONIC, &query_end) == 0);
    if (expected_result == 0) {
        assert(host_pid == expected_pid);
    } else {
        assert(host_pid == 0);
        assert(errno == expected_errno);
    }

    close(listener);
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 0);
    assert(unlink(socket_path) == 0);
    assert(rmdir(directory) == 0);
    return (int64_t)(query_end.tv_sec - query_begin.tv_sec) * 1000 +
           (int64_t)(query_end.tv_nsec - query_begin.tv_nsec) / 1000000;
}

static void test_protocol(void) {
    run_query_case(serve_success, 0, 0, 43210);
    run_query_case(serve_bad_magic, -1, EPROTO, 0);
    run_query_case(serve_bad_version, -1, EPROTO, 0);
    run_query_case(serve_error_status, -1, EPROTO, 0);
    run_query_case(serve_zero_pid, -1, ERANGE, 0);
    run_query_case(serve_large_pid, -1, ERANGE, 0);
    run_query_case(serve_partial_response, -1, ECONNRESET, 0);
}

static void test_absolute_timeout(void) {
    int64_t elapsed_ms;

    elapsed_ms = run_query_case(serve_trickle_response, -1,
                                ETIMEDOUT, 0);
    assert(elapsed_ms >= 80);
    assert(elapsed_ms < 300);
}

static void test_missing_socket(void) {
    pid_t host_pid = 99;

    errno = 0;
    assert(hostpid_broker_query("/tmp/hami-hostpid-does-not-exist",
                                &host_pid) == -1);
    assert(host_pid == 0);
    assert(errno == ENOENT);

    host_pid = 99;
    errno = 0;
    assert(hostpid_broker_query(NULL, &host_pid) == -1);
    assert(host_pid == 0);
    assert(errno == EINVAL);

    host_pid = 99;
    errno = 0;
    assert(hostpid_broker_query("", &host_pid) == -1);
    assert(host_pid == 0);
    assert(errno == EINVAL);

    char long_path[sizeof(((struct sockaddr_un *)0)->sun_path) + 2];
    memset(long_path, 'a', sizeof(long_path));
    long_path[0] = '/';
    long_path[sizeof(long_path) - 1] = '\0';
    host_pid = 99;
    errno = 0;
    assert(hostpid_broker_query(long_path, &host_pid) == -1);
    assert(host_pid == 0);
    assert(errno == ENAMETOOLONG);
}

static void test_enable_gate(void) {
    assert(hostpid_broker_enabled("1") == 1);
    assert(hostpid_broker_enabled(NULL) == 0);
    assert(hostpid_broker_enabled("") == 0);
    assert(hostpid_broker_enabled("0") == 0);
    assert(hostpid_broker_enabled("true") == 0);
    assert(hostpid_broker_enabled("false") == 0);
    assert(hostpid_broker_enabled("01") == 0);
    assert(hostpid_broker_enabled(" 1") == 0);
}

static void test_trust_validation(void) {
    char directory[PATH_MAX];
    char directory_link[PATH_MAX];
    char linked_socket_path[PATH_MAX];
    char socket_path[PATH_MAX];
    int written;
    int listener = make_listener(socket_path, sizeof(socket_path),
                                 directory, sizeof(directory));
    uid_t owner = geteuid();

    assert(chmod(directory, 0700) == 0);
    assert(hostpid_broker_validate_trust(socket_path, owner, 0) == 0);

    errno = 0;
    assert(hostpid_broker_validate_trust(socket_path, owner + 1, 0) == -1);
    assert(errno == EPERM);

    assert(chmod(directory, 0777) == 0);
    errno = 0;
    assert(hostpid_broker_validate_trust(socket_path, owner, 0) == -1);
    assert(errno == EPERM);

    assert(chmod(directory, 0700) == 0);
    errno = 0;
    assert(hostpid_broker_validate_trust(socket_path, owner, 1) == -1);
    assert(errno == EPERM);

    written = snprintf(directory_link, sizeof(directory_link), "%s-link",
                       directory);
    assert(written > 0 && (size_t)written < sizeof(directory_link));
    written = snprintf(linked_socket_path, sizeof(linked_socket_path),
                       "%s/broker.sock", directory_link);
    assert(written > 0 && (size_t)written < sizeof(linked_socket_path));
    assert(symlink(directory, directory_link) == 0);
    errno = 0;
    assert(hostpid_broker_validate_trust(linked_socket_path, owner, 0) == -1);
    assert(errno == ENOTSOCK);
    assert(unlink(directory_link) == 0);

    close(listener);
    assert(unlink(socket_path) == 0);

    int file = open(socket_path, O_CREAT | O_WRONLY, 0600);
    assert(file >= 0);
    close(file);
    errno = 0;
    assert(hostpid_broker_validate_trust(socket_path, owner, 0) == -1);
    assert(errno == ENOTSOCK);
    assert(unlink(socket_path) == 0);

    assert(symlink(directory, socket_path) == 0);
    errno = 0;
    assert(hostpid_broker_validate_trust(socket_path, owner, 0) == -1);
    assert(errno == ENOTSOCK);
    assert(unlink(socket_path) == 0);
    assert(rmdir(directory) == 0);

    errno = 0;
    assert(hostpid_broker_query_trusted("/tmp/other.sock", NULL) == -1);
    assert(errno == EINVAL);

    errno = 0;
    assert(hostpid_broker_validate_trust("relative.sock", owner, 0) == -1);
    assert(errno == EINVAL);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    test_protocol();
    test_absolute_timeout();
    test_missing_socket();
    test_enable_gate();
    test_trust_validation();
    puts("host PID broker client tests passed");
    return 0;
}

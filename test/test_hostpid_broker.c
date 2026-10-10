/* Every check here is an assert, so keep them in release builds too. */
#undef NDEBUG
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
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

enum {
    kOverlongSocketPathSize =
        sizeof(((struct sockaddr_un *)0)->sun_path) + 2,
};

typedef void (*server_action_t)(int connection);

#ifdef HOSTPID_BROKER_TESTING
static atomic_int connect_retry_count;
static atomic_int connect_retry_fd;
static int connect_retry_notify = -1;

void hostpid_broker_test_connect_retry(int fd) {
    atomic_store(&connect_retry_fd, fd);
    if (atomic_fetch_add(&connect_retry_count, 1) == 0 &&
        connect_retry_notify >= 0) {
        assert(write(connect_retry_notify, "r", 1) == 1);
    }
}
#endif

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

static void serve_split_success(int connection) {
    unsigned char response[RESPONSE_SIZE];
    const struct timespec delay = {.tv_sec = 0, .tv_nsec = 1000000L};

    read_request(connection);
    make_response(response, 1, 0, 43210);
    for (size_t i = 0; i < sizeof(response); i++) {
        write_all(connection, response + i, 1);
        nanosleep(&delay, NULL);
    }
}

static void serve_silent(int connection) {
    unsigned char byte;

    read_request(connection);
    assert(read(connection, &byte, 1) == 0);
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
    int written;

    assert(directory_size >= sizeof("/tmp/hami-hostpid-test-XXXXXX"));
    written = snprintf(directory, directory_size,
                       "/tmp/hami-hostpid-test-XXXXXX");
    if (written < 0 || (size_t)written >= directory_size) {
        abort();
    }
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
        alarm(5);
        int connection = accept(listener, NULL, NULL);
        assert(connection >= 0);
        action(connection);
        close(connection);
        close(listener);
        _exit(0);
    }
    close(listener);

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
    run_query_case(serve_split_success, 0, 0, 43210);
    run_query_case(serve_bad_magic, -1, EPROTO, 0);
    run_query_case(serve_bad_version, -1, EPROTO, 0);
    run_query_case(serve_error_status, -1, EPROTO, 0);
    run_query_case(serve_zero_pid, -1, ERANGE, 0);
    run_query_case(serve_large_pid, -1, ERANGE, 0);
    run_query_case(serve_partial_response, -1, ECONNRESET, 0);
    run_query_case(serve_silent, -1, ETIMEDOUT, 0);
}

static void *query_in_thread(void *argument) {
    pid_t host_pid = 99;
    hostpid_broker_query(argument, &host_pid);
    return NULL;
}

#if defined(__linux__) && defined(HOSTPID_BROKER_TESTING)
static int fill_accept_queue(int listener, const char *socket_path) {
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    size_t length = strlen(socket_path);
    assert(length < sizeof(address.sun_path));
    memcpy(address.sun_path, socket_path, length + 1);
    socklen_t address_length =
        (socklen_t)(offsetof(struct sockaddr_un, sun_path) + length + 1);

    /* Linux allows one queued connection with listen(..., 0). */
    assert(listen(listener, 0) == 0);
    int queued = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    assert(queued >= 0);
    assert(connect(queued, (struct sockaddr *)&address, address_length) == 0);
    int probe = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    assert(probe >= 0);
    assert(connect(probe, (struct sockaddr *)&address, address_length) == -1);
    assert(errno == EAGAIN);
    close(probe);
    atomic_store(&connect_retry_count, 0);
    atomic_store(&connect_retry_fd, -1);
    return queued;
}

static void assert_retry_socket_closed(void) {
    assert(atomic_load(&connect_retry_count) > 0);
    int fd = atomic_load(&connect_retry_fd);
    assert(fd >= 0);
    errno = 0;
    assert(fcntl(fd, F_GETFD) == -1);
    assert(errno == EBADF);
}

static void test_full_accept_queue_timeout(void) {
    char directory[PATH_MAX];
    char socket_path[PATH_MAX];
    struct timespec begin, end;
    pid_t host_pid = 99;
    int listener = make_listener(socket_path, sizeof(socket_path),
                                 directory, sizeof(directory));
    int queued = fill_accept_queue(listener, socket_path);

    assert(clock_gettime(CLOCK_MONOTONIC, &begin) == 0);
    assert(hostpid_broker_query(socket_path, &host_pid) == -1);
    assert(errno == ETIMEDOUT);
    assert(host_pid == 0);
    assert(clock_gettime(CLOCK_MONOTONIC, &end) == 0);
    int64_t elapsed_ms = (int64_t)(end.tv_sec - begin.tv_sec) * 1000 +
                        (end.tv_nsec - begin.tv_nsec) / 1000000;
    assert(elapsed_ms >= 80 && elapsed_ms < 300);
    assert(atomic_load(&connect_retry_count) > 1);
    assert_retry_socket_closed();
    close(queued);
    close(listener);
    assert(unlink(socket_path) == 0);
    assert(rmdir(directory) == 0);
    puts("full accept queue times out under the original deadline");
}

static void test_full_accept_queue_recovers(void) {
    char directory[PATH_MAX];
    char socket_path[PATH_MAX];
    int notify[2];
    int status;
    pid_t host_pid = 99;
    int listener = make_listener(socket_path, sizeof(socket_path),
                                 directory, sizeof(directory));
    int queued = fill_accept_queue(listener, socket_path);
    assert(pipe(notify) == 0);
    connect_retry_notify = notify[1];
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        char byte;
        alarm(5);
        close(notify[1]);
        close(queued);
        /* Do not drain the queue until the client really sees EAGAIN. */
        assert(read(notify[0], &byte, 1) == 1);
        close(notify[0]);
        int connection = accept(listener, NULL, NULL);
        assert(connection >= 0);
        close(connection);
        connection = accept(listener, NULL, NULL);
        assert(connection >= 0);
        serve_success(connection);
        close(connection);
        close(listener);
        _exit(0);
    }
    close(notify[0]);
    close(listener);
    assert(hostpid_broker_query(socket_path, &host_pid) == 0);
    assert(host_pid == 43210);
    assert_retry_socket_closed();
    connect_retry_notify = -1;
    close(notify[1]);
    close(queued);
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(unlink(socket_path) == 0);
    assert(rmdir(directory) == 0);
    puts("full accept queue recovers after a confirmed EAGAIN");
}

static void test_cancelled_connect_retry(void) {
    char directory[PATH_MAX];
    char socket_path[PATH_MAX];
    int notify[2];
    char byte;
    pthread_t thread;
    void *result;
    int listener = make_listener(socket_path, sizeof(socket_path),
                                 directory, sizeof(directory));
    int queued = fill_accept_queue(listener, socket_path);
    assert(pipe(notify) == 0);
    connect_retry_notify = notify[1];
    assert(pthread_create(&thread, NULL, query_in_thread, socket_path) == 0);
    assert(read(notify[0], &byte, 1) == 1);
    assert(pthread_cancel(thread) == 0);
    assert(pthread_join(thread, &result) == 0);
    assert(result == PTHREAD_CANCELED);
    assert_retry_socket_closed();
    connect_retry_notify = -1;
    close(notify[0]);
    close(notify[1]);
    close(queued);
    close(listener);
    assert(unlink(socket_path) == 0);
    assert(rmdir(directory) == 0);
    puts("cancelled connect retry closes its socket");
}
#endif

static void test_cancelled_query_closes_socket(void) {
    char directory[PATH_MAX];
    char socket_path[PATH_MAX];
    pthread_t thread;
    void *result;
    unsigned char byte;
    int listener = make_listener(socket_path, sizeof(socket_path),
                                 directory, sizeof(directory));

    assert(pthread_create(&thread, NULL, query_in_thread, socket_path) == 0);
    int connection = accept(listener, NULL, NULL);
    assert(connection >= 0);
    read_request(connection);
    assert(pthread_cancel(thread) == 0);
    assert(pthread_join(thread, &result) == 0);
    assert(result == PTHREAD_CANCELED);
    /* A cancelled query must close its end, so the server observes EOF. */
    assert(recv(connection, &byte, 1, MSG_DONTWAIT) == 0);
    close(connection);
    close(listener);
    assert(unlink(socket_path) == 0);
    assert(rmdir(directory) == 0);
    run_query_case(serve_success, 0, 0, 43210);
}

#ifdef HOSTPID_BROKER_TESTING
static void test_peer_credentials(void) {
    char directory[PATH_MAX];
    char socket_path[PATH_MAX];
    int listener = make_listener(socket_path, sizeof(socket_path),
                                 directory, sizeof(directory));
    pid_t child = fork();
    int status;
    pid_t host_pid = 99;

    assert(child >= 0);
    if (child == 0) {
        unsigned char byte;
        alarm(5);
        int connection = accept(listener, NULL, NULL);
        assert(connection >= 0);
        /* A rejected peer must not receive even the request header. */
        assert(read(connection, &byte, 1) == 0);
        close(connection);
#ifdef __linux__
        connection = accept(listener, NULL, NULL);
        assert(connection >= 0);
        serve_success(connection);
        close(connection);
#endif
        close(listener);
        _exit(0);
    }
    close(listener);
    errno = 0;
    assert(hostpid_broker_test_query_peer(socket_path, &host_pid,
                                          geteuid() + 1) == -1);
    assert(host_pid == 0);
#ifdef __linux__
    assert(errno == EPERM);
    assert(hostpid_broker_test_query_peer(socket_path, &host_pid,
                                          geteuid()) == 0);
    assert(host_pid == 43210);
    puts("peer credential rejection and acceptance passed");
#else
    assert(errno == ENOTSUP);
    puts("SKIP: Linux SO_PEERCRED; unsupported peer checks fail closed");
#endif
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(unlink(socket_path) == 0);
    assert(rmdir(directory) == 0);
}
#endif

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

    char long_path[kOverlongSocketPathSize];
    memset(long_path, 'a', sizeof(long_path));
    long_path[0] = '/';
    long_path[sizeof(long_path) - 1] = '\0';
    host_pid = 99;
    errno = 0;
    assert(hostpid_broker_query(long_path, &host_pid) == -1);
    assert(host_pid == 0);
    assert(errno == ENAMETOOLONG);

    assert(hostpid_broker_query("/tmp/unused.sock", NULL) == -1);
    assert(errno == EINVAL);
    host_pid = 99;
    assert(hostpid_broker_query_trusted("/tmp/other.sock", &host_pid) == -1);
    assert(errno == EINVAL && host_pid == 0);
    host_pid = 99;
    assert(hostpid_broker_query_trusted(NULL, &host_pid) == -1);
    assert(errno == EINVAL && host_pid == 0);
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
    alarm(15);
    signal(SIGPIPE, SIG_IGN);
#if defined(__linux__) && defined(HOSTPID_BROKER_TESTING)
    test_full_accept_queue_timeout();
    test_full_accept_queue_recovers();
    test_cancelled_connect_retry();
#else
    puts("SKIP: Linux Unix socket accept queue saturation");
#endif
    test_cancelled_query_closes_socket();
    test_protocol();
    test_absolute_timeout();
    test_missing_socket();
    test_enable_gate();
    test_trust_validation();
#ifdef HOSTPID_BROKER_TESTING
    test_peer_credentials();
#endif
    puts("host PID broker client tests passed");
    return 0;
}

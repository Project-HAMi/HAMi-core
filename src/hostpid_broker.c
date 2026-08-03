/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#include "include/hostpid_broker.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define HOSTPID_PROTOCOL_VERSION 1
#define HOSTPID_COMMAND_GET_PID 1
#define HOSTPID_STATUS_OK 0
#define HOSTPID_REQUEST_SIZE 8
#define HOSTPID_RESPONSE_SIZE 12

#ifndef HOSTPID_TIMEOUT_MS
#define HOSTPID_TIMEOUT_MS 500
#endif

static const unsigned char hostpid_magic[4] = {'H', 'P', 'I', 'D'};

static int deadline_after_ms(struct timespec *deadline, long timeout_ms) {
    if (clock_gettime(CLOCK_MONOTONIC, deadline) != 0) {
        return -1;
    }
    deadline->tv_sec += timeout_ms / 1000;
    deadline->tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (deadline->tv_nsec >= 1000000000L) {
        deadline->tv_sec++;
        deadline->tv_nsec -= 1000000000L;
    }
    return 0;
}

static int remaining_ms(const struct timespec *deadline) {
    struct timespec now;
    int64_t nanoseconds;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1;
    }
    nanoseconds = (int64_t)(deadline->tv_sec - now.tv_sec) * 1000000000LL +
                  deadline->tv_nsec - now.tv_nsec;
    if (nanoseconds <= 0) {
        errno = ETIMEDOUT;
        return -1;
    }
    nanoseconds = (nanoseconds + 999999LL) / 1000000LL;
    if (nanoseconds > INT_MAX) {
        return INT_MAX;
    }
    return (int)nanoseconds;
}

static int wait_for_fd(int fd, short events,
                       const struct timespec *deadline) {
    struct pollfd descriptor = {
        .fd = fd,
        .events = events,
    };

    for (;;) {
        int timeout_ms = remaining_ms(deadline);
        int result;

        if (timeout_ms < 0) {
            return -1;
        }
        result = poll(&descriptor, 1, timeout_ms);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result == 0) {
            errno = ETIMEDOUT;
            return -1;
        }
        if (result < 0) {
            return -1;
        }
        if ((descriptor.revents & POLLNVAL) != 0) {
            errno = EBADF;
            return -1;
        }
        if ((descriptor.revents & (events | POLLERR | POLLHUP)) != 0) {
            return 0;
        }
    }
}

static int connect_before_deadline(int fd,
                                   const struct sockaddr_un *address,
                                   socklen_t address_length,
                                   const struct timespec *deadline) {
    int result = connect(fd, (const struct sockaddr *)address,
                         address_length);
    int socket_error = 0;
    socklen_t error_length = sizeof(socket_error);

    if (result == 0) {
        return 0;
    }
    if (errno != EINPROGRESS && errno != EAGAIN) {
        return -1;
    }
    if (wait_for_fd(fd, POLLOUT, deadline) != 0) {
        return -1;
    }
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error,
                   &error_length) != 0) {
        return -1;
    }
    if (socket_error != 0) {
        errno = socket_error;
        return -1;
    }
    return 0;
}

static int write_before_deadline(int fd, const unsigned char *buffer,
                                 size_t length,
                                 const struct timespec *deadline) {
    while (length > 0) {
        ssize_t written = send(fd, buffer, length,
                               MSG_DONTWAIT | MSG_NOSIGNAL);

        if (written > 0) {
            buffer += written;
            length -= (size_t)written;
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (wait_for_fd(fd, POLLOUT, deadline) == 0) {
                continue;
            }
            return -1;
        }
        if (written == 0) {
            errno = EPIPE;
        }
        return -1;
    }
    return 0;
}

static int read_before_deadline(int fd, unsigned char *buffer, size_t length,
                                const struct timespec *deadline) {
    while (length > 0) {
        ssize_t received = recv(fd, buffer, length, MSG_DONTWAIT);

        if (received > 0) {
            buffer += received;
            length -= (size_t)received;
            continue;
        }
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (wait_for_fd(fd, POLLIN, deadline) == 0) {
                continue;
            }
            return -1;
        }
        if (received == 0) {
            errno = ECONNRESET;
        }
        return -1;
    }
    return 0;
}

static uint16_t read_u16(const unsigned char *buffer) {
    return (uint16_t)((uint16_t)buffer[0] << 8 | buffer[1]);
}

static uint32_t read_u32(const unsigned char *buffer) {
    return (uint32_t)buffer[0] << 24 |
           (uint32_t)buffer[1] << 16 |
           (uint32_t)buffer[2] << 8 |
           (uint32_t)buffer[3];
}

static int socket_directory(const char *socket_path, char *directory,
                            size_t directory_size) {
    const char *last_slash;
    size_t directory_length;

    if (socket_path == NULL || socket_path[0] != '/') {
        errno = EINVAL;
        return -1;
    }
    last_slash = strrchr(socket_path, '/');
    if (last_slash == socket_path) {
        errno = EINVAL;
        return -1;
    }
    directory_length = (size_t)(last_slash - socket_path);
    if (directory_length >= directory_size) {
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(directory, socket_path, directory_length);
    directory[directory_length] = '\0';
    return 0;
}

int hostpid_broker_validate_trust(const char *socket_path,
                                  uid_t trusted_uid,
                                  int require_readonly) {
    struct stat socket_stat;
    struct stat directory_stat;
    struct statvfs filesystem_stat;
    char directory[sizeof(((struct sockaddr_un *)0)->sun_path)];

    if (socket_directory(socket_path, directory, sizeof(directory)) != 0) {
        return -1;
    }
    if (lstat(directory, &directory_stat) != 0 ||
        lstat(socket_path, &socket_stat) != 0) {
        return -1;
    }
    if (!S_ISDIR(directory_stat.st_mode) ||
        !S_ISSOCK(socket_stat.st_mode)) {
        errno = ENOTSOCK;
        return -1;
    }
    if (directory_stat.st_uid != trusted_uid ||
        socket_stat.st_uid != trusted_uid ||
        (directory_stat.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        errno = EPERM;
        return -1;
    }
    if (require_readonly) {
        if (statvfs(directory, &filesystem_stat) != 0) {
            return -1;
        }
        if ((filesystem_stat.f_flag & ST_RDONLY) == 0) {
            errno = EPERM;
            return -1;
        }
    }
    return 0;
}

static int verify_peer_uid(int fd, uid_t trusted_uid) {
#ifdef SO_PEERCRED
    struct ucred credentials;
    socklen_t credentials_length = sizeof(credentials);

    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials,
                   &credentials_length) != 0) {
        return -1;
    }
    if (credentials_length != sizeof(credentials) ||
        credentials.uid != trusted_uid) {
        errno = EPERM;
        return -1;
    }
    return 0;
#else
    (void)fd;
    (void)trusted_uid;
    errno = ENOTSUP;
    return -1;
#endif
}

static int query_broker(const char *socket_path, pid_t *host_pid,
                        int verify_peer) {
    static const unsigned char request[HOSTPID_REQUEST_SIZE] = {
        'H', 'P', 'I', 'D',
        0, HOSTPID_PROTOCOL_VERSION,
        0, HOSTPID_COMMAND_GET_PID,
    };
    unsigned char response[HOSTPID_RESPONSE_SIZE];
    struct sockaddr_un address;
    struct timespec deadline;
    size_t path_length;
    int fd;
    int saved_errno;

    if (host_pid == NULL) {
        errno = EINVAL;
        return -1;
    }
    *host_pid = 0;
    if (socket_path == NULL) {
        errno = EINVAL;
        return -1;
    }
    path_length = strlen(socket_path);
    if (path_length == 0 || path_length >= sizeof(address.sun_path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    if (deadline_after_ms(&deadline, HOSTPID_TIMEOUT_MS) != 0) {
        return -1;
    }

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(fd, F_SETFL, O_NONBLOCK) != 0) {
        saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, socket_path, path_length + 1);
    socklen_t address_length =
        (socklen_t)(offsetof(struct sockaddr_un, sun_path) + path_length + 1);
#ifdef __APPLE__
    address.sun_len = address_length;
#endif

    if (connect_before_deadline(fd, &address, address_length, &deadline) != 0 ||
        (verify_peer && verify_peer_uid(fd, 0) != 0) ||
        write_before_deadline(fd, request, sizeof(request), &deadline) != 0 ||
        read_before_deadline(fd, response, sizeof(response), &deadline) != 0) {
        saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }
    close(fd);

    if (memcmp(response, hostpid_magic, sizeof(hostpid_magic)) != 0 ||
        read_u16(response + 4) != HOSTPID_PROTOCOL_VERSION ||
        read_u16(response + 6) != HOSTPID_STATUS_OK) {
        errno = EPROTO;
        return -1;
    }
    uint32_t pid = read_u32(response + 8);
    if (pid == 0 || pid > INT_MAX) {
        errno = ERANGE;
        return -1;
    }
    *host_pid = (pid_t)pid;
    return 0;
}

int hostpid_broker_query(const char *socket_path, pid_t *host_pid) {
    return query_broker(socket_path, host_pid, 0);
}

int hostpid_broker_query_trusted(const char *socket_path, pid_t *host_pid) {
    if (host_pid == NULL) {
        errno = EINVAL;
        return -1;
    }
    *host_pid = 0;
    if (socket_path == NULL ||
        strcmp(socket_path, HOSTPID_BROKER_SOCKET_PATH) != 0) {
        errno = EINVAL;
        return -1;
    }
    if (hostpid_broker_validate_trust(socket_path, 0, 1) != 0) {
        return -1;
    }
    return query_broker(socket_path, host_pid, 1);
}

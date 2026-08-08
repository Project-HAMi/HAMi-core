/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#include "include/hostpid_fallback_lock.h"

#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#ifndef HOSTPID_FALLBACK_LOCK_TIMEOUT_MS
#define HOSTPID_FALLBACK_LOCK_TIMEOUT_MS 30000U
#endif

#define HOSTPID_FALLBACK_LOCK_INITIAL_RETRY_US 1000U
#define HOSTPID_FALLBACK_LOCK_MAX_RETRY_US 100000U

static _Atomic int active_lock_fd = -1;

static int monotonic_now(struct timespec *now) {
    if (clock_gettime(CLOCK_MONOTONIC, now) != 0) {
        return -1;
    }
    return 0;
}

static int deadline_after_ms(struct timespec *deadline,
                             unsigned int timeout_ms) {
    if (monotonic_now(deadline) != 0) {
        return -1;
    }
    deadline->tv_sec += timeout_ms / 1000U;
    deadline->tv_nsec += (long)(timeout_ms % 1000U) * 1000000L;
    if (deadline->tv_nsec >= 1000000000L) {
        deadline->tv_sec++;
        deadline->tv_nsec -= 1000000000L;
    }
    return 0;
}

static int deadline_expired(const struct timespec *deadline) {
    struct timespec now;

    if (monotonic_now(&now) != 0) {
        return -1;
    }
    if (now.tv_sec > deadline->tv_sec ||
        (now.tv_sec == deadline->tv_sec &&
         now.tv_nsec >= deadline->tv_nsec)) {
        errno = ETIMEDOUT;
        return 1;
    }
    return 0;
}

static int sleep_before_retry(unsigned int delay_us) {
    struct timespec delay = {
        .tv_sec = delay_us / 1000000U,
        .tv_nsec = (long)(delay_us % 1000000U) * 1000L,
    };

    while (nanosleep(&delay, &delay) != 0) {
        if (errno != EINTR) {
            return -1;
        }
    }
    return 0;
}

static int validate_lock_object(int fd, const char *path,
                                uid_t trusted_owner,
                                const struct stat *opened_stat,
                                int require_readonly) {
    struct stat current_stat;
    struct statvfs filesystem_stat;

    if (!S_ISDIR(opened_stat->st_mode) ||
        opened_stat->st_uid != trusted_owner) {
        errno = EACCES;
        return -1;
    }
    if (fstatat(AT_FDCWD, path, &current_stat, AT_SYMLINK_NOFOLLOW) != 0) {
        return -1;
    }
    if (!S_ISDIR(current_stat.st_mode) ||
        current_stat.st_uid != trusted_owner ||
        current_stat.st_dev != opened_stat->st_dev ||
        current_stat.st_ino != opened_stat->st_ino) {
        errno = ESTALE;
        return -1;
    }
    if (fcntl(fd, F_GETFD) < 0) {
        return -1;
    }
    if (require_readonly) {
        if (fstatvfs(fd, &filesystem_stat) != 0) {
            return -1;
        }
        if ((filesystem_stat.f_flag & ST_RDONLY) == 0) {
            errno = EACCES;
            return -1;
        }
    }
    return 0;
}

static void discard_active_fd(int fd) {
    int expected = fd;

    atomic_compare_exchange_strong_explicit(&active_lock_fd, &expected, -1,
                                            memory_order_acq_rel,
                                            memory_order_acquire);
    close(fd);
}

static int acquire_at(const char *path, uid_t trusted_owner,
                      unsigned int timeout_ms, int require_readonly) {
    struct timespec deadline;
    struct stat opened_stat;
    unsigned int retry_us = HOSTPID_FALLBACK_LOCK_INITIAL_RETRY_US;
    int expected = -1;
    int descriptor_flags;
    int fd;

    if (path == NULL || path[0] != '/' || timeout_ms == 0U) {
        errno = EINVAL;
        return -1;
    }
    fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        return -1;
    }
    descriptor_flags = fcntl(fd, F_GETFD);
    if (descriptor_flags < 0 ||
        fcntl(fd, F_SETFD, descriptor_flags | FD_CLOEXEC) != 0) {
        int saved_errno = errno;

        close(fd);
        errno = saved_errno;
        return -1;
    }
    if (fstat(fd, &opened_stat) != 0) {
        int saved_errno = errno;

        close(fd);
        errno = saved_errno;
        return -1;
    }
    if (!S_ISDIR(opened_stat.st_mode) ||
        opened_stat.st_uid != trusted_owner) {
        close(fd);
        errno = EACCES;
        return -1;
    }
    if (!atomic_compare_exchange_strong_explicit(
            &active_lock_fd, &expected, fd, memory_order_acq_rel,
            memory_order_acquire)) {
        close(fd);
        errno = EDEADLK;
        return -1;
    }
    if (deadline_after_ms(&deadline, timeout_ms) != 0) {
        int saved_errno = errno;

        discard_active_fd(fd);
        errno = saved_errno;
        return -1;
    }

    for (;;) {
        int expired;

        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            break;
        }
        if (errno != EWOULDBLOCK && errno != EAGAIN && errno != EINTR) {
            int saved_errno = errno;

            discard_active_fd(fd);
            errno = saved_errno;
            return -1;
        }
        expired = deadline_expired(&deadline);
        if (expired != 0) {
            int saved_errno = errno;

            discard_active_fd(fd);
            errno = saved_errno;
            return -1;
        }
        if (sleep_before_retry(retry_us) != 0) {
            int saved_errno = errno;

            discard_active_fd(fd);
            errno = saved_errno;
            return -1;
        }
        if (retry_us < HOSTPID_FALLBACK_LOCK_MAX_RETRY_US / 2U) {
            retry_us *= 2U;
        } else {
            retry_us = HOSTPID_FALLBACK_LOCK_MAX_RETRY_US;
        }
    }

    if (validate_lock_object(fd, path, trusted_owner, &opened_stat,
                             require_readonly) != 0) {
        int saved_errno = errno;

        flock(fd, LOCK_UN);
        discard_active_fd(fd);
        errno = saved_errno;
        return -1;
    }
    return 0;
}

int hostpid_fallback_lock_acquire_at(const char *path, uid_t trusted_owner,
                                     unsigned int timeout_ms) {
    return acquire_at(path, trusted_owner, timeout_ms, 0);
}

int hostpid_fallback_lock_acquire(void) {
    return acquire_at(HOSTPID_FALLBACK_LOCK_PATH, 0,
                      HOSTPID_FALLBACK_LOCK_TIMEOUT_MS, 1);
}

int hostpid_fallback_lock_release(void) {
    int fd = atomic_exchange_explicit(&active_lock_fd, -1,
                                      memory_order_acq_rel);
    int result = 0;
    int saved_errno = 0;

    if (fd < 0) {
        errno = ENOLCK;
        return -1;
    }
    while (flock(fd, LOCK_UN) != 0) {
        if (errno != EINTR) {
            result = -1;
            saved_errno = errno;
            break;
        }
    }
    if (close(fd) != 0 && result == 0) {
        result = -1;
        saved_errno = errno;
    }
    if (result != 0) {
        errno = saved_errno;
    }
    return result;
}

void hostpid_fallback_lock_after_fork(void) {
    int fd = atomic_exchange_explicit(&active_lock_fd, -1,
                                      memory_order_acq_rel);

    if (fd >= 0) {
        close(fd);
    }
}

int hostpid_fallback_lock_active_fd(void) {
    return atomic_load_explicit(&active_lock_fd, memory_order_acquire);
}

#include "include/hostpid_fallback_lock.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>
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

int hostpid_fallback_lock_deadline_after_ms(struct timespec *deadline,
                                            unsigned int timeout_ms) {
    if (deadline == NULL || timeout_ms == 0U) {
        errno = EINVAL;
        return -1;
    }
    if (monotonic_now(deadline) != 0) {
        return -1;
    }
    deadline->tv_sec += timeout_ms / 1000U;
    deadline->tv_nsec += (int64_t)(timeout_ms % 1000U) * INT64_C(1000000);
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

static int deadline_remaining(const struct timespec *deadline,
                              struct timespec *remaining) {
    struct timespec now;

    if (monotonic_now(&now) != 0) {
        return -1;
    }
    if (now.tv_sec > deadline->tv_sec ||
        (now.tv_sec == deadline->tv_sec &&
         now.tv_nsec >= deadline->tv_nsec)) {
        errno = ETIMEDOUT;
        return -1;
    }
    remaining->tv_sec = deadline->tv_sec - now.tv_sec;
    remaining->tv_nsec = deadline->tv_nsec - now.tv_nsec;
    if (remaining->tv_nsec < 0) {
        remaining->tv_sec--;
        remaining->tv_nsec += 1000000000L;
    }
    return 0;
}

static int sleep_before_retry(unsigned int delay_us,
                              const struct timespec *deadline) {
    struct timespec delay = {
        .tv_sec = delay_us / 1000000U,
        .tv_nsec = (int64_t)(delay_us % 1000000U) * INT64_C(1000),
    };
    struct timespec remaining;

    if (deadline_remaining(deadline, &remaining) != 0) {
        return -1;
    }
    if (delay.tv_sec > remaining.tv_sec ||
        (delay.tv_sec == remaining.tv_sec &&
         delay.tv_nsec > remaining.tv_nsec)) {
        delay = remaining;
    }

    while (nanosleep(&delay, &delay) != 0) {
        if (errno != EINTR) {
            return -1;
        }
        if (deadline_remaining(deadline, &remaining) != 0) {
            return -1;
        }
        if (delay.tv_sec > remaining.tv_sec ||
            (delay.tv_sec == remaining.tv_sec &&
             delay.tv_nsec > remaining.tv_nsec)) {
            delay = remaining;
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

static int acquire_at_until(const char *path, uid_t trusted_owner,
                            const struct timespec *deadline,
                            int require_readonly) {
    unsigned int retry_us = HOSTPID_FALLBACK_LOCK_INITIAL_RETRY_US;
    int expected = -1;
    int fd;

    (void)trusted_owner;
    (void)require_readonly;
    if (path == NULL || path[0] != '/' || deadline == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (deadline_expired(deadline) != 0) {
        return -1;
    }
    fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }
    if (!atomic_compare_exchange_strong_explicit(
            &active_lock_fd, &expected, fd, memory_order_acq_rel,
            memory_order_acquire)) {
        close(fd);
        errno = EDEADLK;
        return -1;
    }
    for (;;) {
        int expired;

        expired = deadline_expired(deadline);
        if (expired != 0) {
            int saved_errno = errno;

            discard_active_fd(fd);
            errno = saved_errno;
            return -1;
        }
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            expired = deadline_expired(deadline);
            if (expired == 0) {
                break;
            }
            int saved_errno = errno;

            discard_active_fd(fd);
            errno = saved_errno;
            return -1;
        }
        if (errno != EWOULDBLOCK && errno != EAGAIN && errno != EINTR) {
            int saved_errno = errno;

            discard_active_fd(fd);
            errno = saved_errno;
            return -1;
        }
        if (sleep_before_retry(retry_us, deadline) != 0) {
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
    return 0;
}

int hostpid_fallback_lock_acquire_at(const char *path, uid_t trusted_owner,
                                     unsigned int timeout_ms) {
    struct timespec deadline;

    if (hostpid_fallback_lock_deadline_after_ms(&deadline, timeout_ms) != 0) {
        return -1;
    }
    return acquire_at_until(path, trusted_owner, &deadline, 0);
}

int hostpid_fallback_lock_acquire_at_until(
    const char *path, uid_t trusted_owner, const struct timespec *deadline) {
    return acquire_at_until(path, trusted_owner, deadline, 0);
}

int hostpid_fallback_lock_acquire_until(const struct timespec *deadline) {
    return acquire_at_until(HOSTPID_FALLBACK_LOCK_PATH, 0, deadline, 1);
}

int hostpid_fallback_lock_acquire(void) {
    struct timespec deadline;

    if (hostpid_fallback_lock_deadline_after_ms(
            &deadline, HOSTPID_FALLBACK_LOCK_TIMEOUT_MS) != 0) {
        return -1;
    }
    return hostpid_fallback_lock_acquire_until(&deadline);
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

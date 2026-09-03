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

#ifdef HOSTPID_FALLBACK_LOCK_TESTING
static hostpid_fallback_lock_test_hook before_flock_hook;

void hostpid_fallback_lock_set_before_flock_hook(
    hostpid_fallback_lock_test_hook hook) {
    before_flock_hook = hook;
}
#endif

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

static int validate_directory_metadata(const struct stat *directory_stat,
                                       uid_t trusted_owner) {
    mode_t writable = S_IWGRP | S_IWOTH;

    if (!S_ISDIR(directory_stat->st_mode) ||
        directory_stat->st_uid != trusted_owner) {
        errno = EACCES;
        return -1;
    }
    if ((directory_stat->st_mode & writable) != 0 &&
        (directory_stat->st_mode & S_ISVTX) == 0) {
        errno = EACCES;
        return -1;
    }
    return 0;
}

#ifdef __linux__
static int validate_supported_filesystem(int fd);

static int validate_path_component(int fd,
                                   const struct stat *component_stat,
                                   uid_t trusted_owner) {
    if (validate_directory_metadata(component_stat, trusted_owner) != 0) {
        return -1;
    }
    return validate_supported_filesystem(fd);
}
#endif

static int open_directory_without_symlinks(const char *path,
                                           uid_t trusted_owner,
                                           int validate_components) {
#ifdef __linux__
    const char *cursor;
    int directory_fd;

    if (path == NULL || path[0] != '/') {
        errno = EINVAL;
        return -1;
    }
    directory_fd = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory_fd < 0) {
        return -1;
    }
    if (validate_components) {
        struct stat root_stat;

        if (fstat(directory_fd, &root_stat) != 0 ||
            validate_path_component(directory_fd, &root_stat,
                                    trusted_owner) != 0) {
            int saved_errno = errno;

            close(directory_fd);
            errno = saved_errno;
            return -1;
        }
    }
    cursor = path;
    while (*cursor == '/') {
        cursor++;
    }
    while (*cursor != '\0') {
        char component[NAME_MAX + 1U];
        const char *end = cursor;
        size_t length;
        int next_fd;

        while (*end != '\0' && *end != '/') {
            end++;
        }
        length = (size_t)(end - cursor);
        if (length == 0U || length > NAME_MAX) {
            close(directory_fd);
            errno = ENAMETOOLONG;
            return -1;
        }
        memcpy(component, cursor, length);
        component[length] = '\0';
        if (strcmp(component, ".") == 0 || strcmp(component, "..") == 0) {
            close(directory_fd);
            errno = EINVAL;
            return -1;
        }
        next_fd = openat(directory_fd, component,
                         O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (next_fd < 0) {
            int saved_errno = errno;

            close(directory_fd);
            errno = saved_errno;
            return -1;
        }
        if (validate_components) {
            struct stat component_stat;

            if (fstat(next_fd, &component_stat) != 0 ||
                validate_path_component(next_fd, &component_stat,
                                        trusted_owner) != 0) {
                int saved_errno = errno;

                close(next_fd);
                close(directory_fd);
                errno = saved_errno;
                return -1;
            }
        }
        close(directory_fd);
        directory_fd = next_fd;
        cursor = end;
        while (*cursor == '/') {
            cursor++;
        }
    }
    return directory_fd;
#else
    (void)trusted_owner;
    (void)validate_components;
    return open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
#endif
}

#ifdef __linux__
static int filesystem_type_supported(uint64_t filesystem_type) {
#ifdef EXT4_SUPER_MAGIC
    if (filesystem_type == (uint64_t)EXT4_SUPER_MAGIC) {
        return 1;
    }
#endif
#ifdef XFS_SUPER_MAGIC
    if (filesystem_type == (uint64_t)XFS_SUPER_MAGIC) {
        return 1;
    }
#endif
#ifdef TMPFS_MAGIC
    if (filesystem_type == (uint64_t)TMPFS_MAGIC) {
        return 1;
    }
#endif
#ifdef BTRFS_SUPER_MAGIC
    if (filesystem_type == (uint64_t)BTRFS_SUPER_MAGIC) {
        return 1;
    }
#endif
#ifdef F2FS_SUPER_MAGIC
    if (filesystem_type == (uint64_t)F2FS_SUPER_MAGIC) {
        return 1;
    }
#endif
#ifdef OVERLAYFS_SUPER_MAGIC
    if (filesystem_type == (uint64_t)OVERLAYFS_SUPER_MAGIC) {
        return 1;
    }
#endif
#ifdef RAMFS_MAGIC
    if (filesystem_type == (uint64_t)RAMFS_MAGIC) {
        return 1;
    }
#endif
#ifdef ZFS_SUPER_MAGIC
    if (filesystem_type == (uint64_t)ZFS_SUPER_MAGIC) {
        return 1;
    }
#endif
    return 0;
}
#endif

static int validate_supported_filesystem(int fd) {
#ifdef __linux__
    struct statfs filesystem_stat;

    if (fstatfs(fd, &filesystem_stat) != 0) {
        return -1;
    }
    if (filesystem_type_supported((uint64_t)filesystem_stat.f_type)) {
        return 0;
    }
    errno = EOPNOTSUPP;
    return -1;
#else
    (void)fd;
    return 0;
#endif
}

static int validate_readonly_filesystem(int fd) {
    struct statvfs filesystem_stat;

    if (fstatvfs(fd, &filesystem_stat) != 0) {
        return -1;
    }
    if ((filesystem_stat.f_flag & ST_RDONLY) == 0) {
        errno = EACCES;
        return -1;
    }
    return validate_supported_filesystem(fd);
}

static int validate_lock_object(int fd, const char *path,
                                uid_t trusted_owner,
                                const struct stat *opened_stat,
                                int require_readonly) {
    struct stat current_stat;
    struct stat descriptor_stat;
    int current_fd;

    if (validate_directory_metadata(opened_stat, trusted_owner) != 0) {
        return -1;
    }
    if (fstat(fd, &descriptor_stat) != 0) {
        return -1;
    }
    if (descriptor_stat.st_dev != opened_stat->st_dev ||
        descriptor_stat.st_ino != opened_stat->st_ino) {
        errno = ESTALE;
        return -1;
    }
    if (validate_directory_metadata(&descriptor_stat, trusted_owner) != 0) {
        return -1;
    }
    if (fcntl(fd, F_GETFD) < 0) {
        return -1;
    }
    current_fd = open_directory_without_symlinks(path, trusted_owner,
                                                  require_readonly);
    if (current_fd < 0) {
        return -1;
    }
    if (fstat(current_fd, &current_stat) != 0) {
        int saved_errno = errno;

        close(current_fd);
        errno = saved_errno;
        return -1;
    }
    if (current_stat.st_dev != opened_stat->st_dev ||
        current_stat.st_ino != opened_stat->st_ino) {
        close(current_fd);
        errno = ESTALE;
        return -1;
    }
    if (validate_directory_metadata(&current_stat, trusted_owner) != 0) {
        int saved_errno = errno;

        close(current_fd);
        errno = saved_errno;
        return -1;
    }
    if (require_readonly) {
        if (validate_readonly_filesystem(fd) != 0 ||
            validate_readonly_filesystem(current_fd) != 0) {
            int saved_errno = errno;

            close(current_fd);
            errno = saved_errno;
            return -1;
        }
    }
    if (close(current_fd) != 0) {
        return -1;
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
    struct stat opened_stat;
    unsigned int retry_us = HOSTPID_FALLBACK_LOCK_INITIAL_RETRY_US;
    int expected = -1;
    int descriptor_flags;
    int fd;

    if (path == NULL || path[0] != '/' || deadline == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (deadline_expired(deadline) != 0) {
        return -1;
    }
    fd = open_directory_without_symlinks(path, trusted_owner,
                                         require_readonly);
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
    if (validate_lock_object(fd, path, trusted_owner, &opened_stat,
                             require_readonly) != 0 ||
        deadline_expired(deadline) != 0) {
        int saved_errno = errno;

        close(fd);
        errno = saved_errno;
        return -1;
    }
    if (!atomic_compare_exchange_strong_explicit(
            &active_lock_fd, &expected, fd, memory_order_acq_rel,
            memory_order_acquire)) {
        close(fd);
        errno = EDEADLK;
        return -1;
    }
#ifdef HOSTPID_FALLBACK_LOCK_TESTING
    if (before_flock_hook != NULL) {
        before_flock_hook();
    }
#endif
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

    if (validate_lock_object(fd, path, trusted_owner, &opened_stat,
                             require_readonly) != 0 ||
        deadline_expired(deadline) != 0) {
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

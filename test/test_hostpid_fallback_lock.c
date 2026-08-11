/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "../src/include/hostpid_fallback_lock.h"

static int failures;
static int waiter_ready_fd = -1;

static void signal_waiter_ready(void) {
    if (waiter_ready_fd >= 0) {
        char byte = '1';

        if (write(waiter_ready_fd, &byte, 1) != 1) {
            _exit(4);
        }
        waiter_ready_fd = -1;
    }
}

static int parse_unsigned(const char *value, uintmax_t *parsed) {
    char *end = NULL;

    errno = 0;
    *parsed = strtoumax(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' ? 0 : -1;
}

static int probe_lock(const char *path, const char *owner_value,
                      const char *timeout_value, const char *hold_value,
                      int expect_timeout) {
    struct timespec hold;
    uintmax_t owner;
    uintmax_t timeout_ms;
    uintmax_t hold_ms;

    if (parse_unsigned(owner_value, &owner) != 0 ||
        parse_unsigned(timeout_value, &timeout_ms) != 0 ||
        parse_unsigned(hold_value, &hold_ms) != 0 ||
        owner > (uintmax_t)(uid_t)-1 || timeout_ms > UINT_MAX ||
        hold_ms > UINT_MAX) {
        return 2;
    }
    if (hostpid_fallback_lock_acquire_at(path, (uid_t)owner,
                                         (unsigned int)timeout_ms) != 0) {
        if (expect_timeout && errno == ETIMEDOUT) {
            puts("timed_out");
            return 0;
        }
        perror("hostpid_fallback_lock_acquire_at");
        return 3;
    }
    if (expect_timeout) {
        hostpid_fallback_lock_release();
        fputs("unexpected_acquire\n", stderr);
        return 4;
    }
    puts("acquired");
    fflush(stdout);
    hold.tv_sec = (time_t)(hold_ms / UINTMAX_C(1000));
    hold.tv_nsec = (int64_t)(hold_ms % UINTMAX_C(1000)) * INT64_C(1000000);
    while (nanosleep(&hold, &hold) != 0 && errno == EINTR) {
    }
    if (hostpid_fallback_lock_release() != 0) {
        perror("hostpid_fallback_lock_release");
        return 5;
    }
    return 0;
}

static int probe_default_lock(const char *hold_value) {
    struct timespec hold;
    uintmax_t hold_ms;

    if (parse_unsigned(hold_value, &hold_ms) != 0 || hold_ms > UINT_MAX) {
        return 2;
    }
    if (hostpid_fallback_lock_acquire() != 0) {
        fprintf(stderr, "rejected_errno=%d\n", errno);
        perror("hostpid_fallback_lock_acquire");
        return 3;
    }
    puts("acquired");
    fflush(stdout);
    hold.tv_sec = (time_t)(hold_ms / UINTMAX_C(1000));
    hold.tv_nsec = (int64_t)(hold_ms % UINTMAX_C(1000)) * INT64_C(1000000);
    while (nanosleep(&hold, &hold) != 0 && errno == EINTR) {
    }
    if (hostpid_fallback_lock_release() != 0) {
        perror("hostpid_fallback_lock_release");
        return 4;
    }
    return 0;
}

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s (errno=%d: %s)\n", message, errno,
                strerror(errno));
        failures++;
    }
}

static int join_path(char *destination, size_t destination_size,
                     const char *parent, const char *suffix) {
    size_t parent_length = strlen(parent);
    size_t suffix_length = strlen(suffix);

    if (parent_length >= destination_size ||
        suffix_length >= destination_size - parent_length) {
        if (destination_size > 0) {
            destination[0] = '\0';
        }
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(destination, parent, parent_length);
    memcpy(destination + parent_length, suffix, suffix_length + 1);
    return 0;
}

static void test_discovery_path_selection(void) {
    int failures_before = failures;

    check(hostpid_discovery_path_select(0, 1) ==
              HOSTPID_DISCOVERY_BROKER,
          "broker success bypasses disabled fallback");
    check(hostpid_discovery_path_select(1, 1) ==
              HOSTPID_DISCOVERY_BROKER,
          "broker success bypasses enabled fallback");
    check(hostpid_discovery_path_select(0, 0) ==
              HOSTPID_DISCOVERY_CACHE_LOCAL,
          "broker disabled failure selects cache local fallback");
    check(hostpid_discovery_path_select(1, 0) ==
              HOSTPID_DISCOVERY_NODE_WIDE,
          "broker enabled failure selects node wide fallback");
    if (failures == failures_before) {
        puts("broker_success_bypass=passed");
        puts("forced_fallback_selection=passed");
    }
}

static int wait_for_child(pid_t child, int expected_status) {
    int status;

    if (waitpid(child, &status, 0) != child) {
        return -1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != expected_status) {
        return -1;
    }
    return 0;
}

static double monotonic_milliseconds(void) {
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1.0;
    }
    return (double)now.tv_sec * 1000.0 +
           (double)now.tv_nsec / 1000000.0;
}

static int child_timeout(const char *path, unsigned int timeout_ms) {
    double started = monotonic_milliseconds();
    double elapsed;

    if (started < 0.0) {
        return 4;
    }
    hostpid_fallback_lock_after_fork();
    if (hostpid_fallback_lock_acquire_at(path, getuid(), timeout_ms) == 0) {
        hostpid_fallback_lock_release();
        return 2;
    }
    if (errno != ETIMEDOUT) {
        return 3;
    }
    elapsed = monotonic_milliseconds() - started;
    dprintf(STDOUT_FILENO, "timeout_elapsed_ms=%.3f\n", elapsed);
    if (elapsed < (double)timeout_ms / 2.0 ||
        elapsed > (double)timeout_ms + 500.0) {
        return 5;
    }
    return 0;
}

static int child_acquire(const char *path, unsigned int timeout_ms) {
    hostpid_fallback_lock_after_fork();
    if (hostpid_fallback_lock_acquire_at(path, getuid(), timeout_ms) != 0) {
        return 2;
    }
    if (hostpid_fallback_lock_release() != 0) {
        return 3;
    }
    return 0;
}

static int wait_until_child_opened(pid_t child, const char *path) {
    char fd_directory[64];
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 1000000L};
    int attempt;

    snprintf(fd_directory, sizeof(fd_directory), "/proc/%" PRIdMAX "/fd",
             (intmax_t)child);
    for (attempt = 0; attempt < 2000; attempt++) {
        DIR *directory = opendir(fd_directory);

        if (directory != NULL) {
            struct dirent *entry;

            while ((entry = readdir(directory)) != NULL) {
                char fd_path[PATH_MAX];
                char target[4096];
                int path_length;
                ssize_t length;

                if (entry->d_name[0] == '.') {
                    continue;
                }
                path_length = snprintf(fd_path, sizeof(fd_path), "%s/%s",
                                       fd_directory, entry->d_name);
                if (path_length < 0 ||
                    (size_t)path_length >= sizeof(fd_path)) {
                    continue;
                }
                length = readlink(fd_path, target, sizeof(target) - 1U);
                if (length < 0) {
                    continue;
                }
                target[length] = '\0';
                if (strcmp(target, path) == 0) {
                    closedir(directory);
                    return 0;
                }
            }
            closedir(directory);
        }
        nanosleep(&delay, NULL);
    }
    return -1;
}

static void test_absolute_deadline_api(const char *path) {
    struct timespec deadline;
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 5000000L};

    errno = 0;
    check(hostpid_fallback_lock_deadline_after_ms(NULL, 50U) == -1 &&
              errno == EINVAL,
          "null deadline output is rejected");
    errno = 0;
    check(hostpid_fallback_lock_deadline_after_ms(&deadline, 0U) == -1 &&
              errno == EINVAL,
          "zero deadline duration is rejected");
    errno = 0;
    check(hostpid_fallback_lock_acquire_at_until(path, getuid(), NULL) == -1 &&
              errno == EINVAL,
          "null absolute deadline is rejected");

    check(hostpid_fallback_lock_deadline_after_ms(&deadline, 500U) == 0,
          "future absolute deadline is created");
    check(hostpid_fallback_lock_acquire_at_until(path, getuid(),
                                                 &deadline) == 0,
          "absolute deadline acquire succeeds");
    check(hostpid_fallback_lock_release() == 0,
          "absolute deadline acquire releases");

    check(hostpid_fallback_lock_deadline_after_ms(&deadline, 1U) == 0,
          "short absolute deadline is created");
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {
    }
    errno = 0;
    check(hostpid_fallback_lock_acquire_at_until(path, getuid(),
                                                 &deadline) == -1 &&
              errno == ETIMEDOUT,
          "expired absolute deadline is rejected");
}

static void test_basic_contract(const char *path) {
    int fd;

    check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
          "basic acquire");
    fd = hostpid_fallback_lock_active_fd();
    check(fd >= 0, "active descriptor is recorded");
    if (fd >= 0) {
        int flags = fcntl(fd, F_GETFD);

        check(flags >= 0 && (flags & FD_CLOEXEC) != 0,
              "descriptor is close on exec");
    }
    errno = 0;
    check(hostpid_fallback_lock_acquire_at(path, getuid(), 50) == -1 &&
              errno == EDEADLK,
          "recursive acquire fails");
    check(hostpid_fallback_lock_release() == 0, "basic release");
    errno = 0;
    check(hostpid_fallback_lock_release() == -1 && errno == ENOLCK,
          "release without owner fails");
}

static void test_path_trust(const char *path) {
    char file_path[] = "/tmp/hami-hostpid-lock-file.XXXXXX";
    char link_path[] = "/tmp/hami-hostpid-lock-link.XXXXXX";
    char ancestor_directory[] = "/tmp/hami-hostpid-lock-ancestor.XXXXXX";
    char ancestor_link[PATH_MAX];
    char ancestor_nested_target[PATH_MAX];
    char ancestor_target[PATH_MAX];
    char missing_path[] = "/tmp/hami-hostpid-lock-missing.XXXXXX";
    char nested_path[PATH_MAX];
    int file_fd;

    errno = 0;
    check(hostpid_fallback_lock_acquire_at("relative", getuid(), 50) == -1 &&
              errno == EINVAL,
          "relative path is rejected");

    file_fd = mkstemp(missing_path);
    check(file_fd >= 0, "missing path fixture name");
    if (file_fd >= 0) {
        close(file_fd);
        unlink(missing_path);
        errno = 0;
        check(hostpid_fallback_lock_acquire_at(missing_path, getuid(), 50) ==
                  -1 && errno == ENOENT,
              "missing path is rejected");
    }

    file_fd = mkstemp(file_path);
    check(file_fd >= 0, "regular file fixture");
    if (file_fd >= 0) {
        close(file_fd);
        check(hostpid_fallback_lock_acquire_at(file_path, getuid(), 50) == -1,
              "regular file is rejected");
        unlink(file_path);
    }

    file_fd = mkstemp(link_path);
    check(file_fd >= 0, "symlink fixture name");
    if (file_fd >= 0) {
        close(file_fd);
        unlink(link_path);
        check(symlink(path, link_path) == 0, "symlink fixture");
        check(hostpid_fallback_lock_acquire_at(link_path, getuid(), 50) == -1,
              "symlink is rejected");
        unlink(link_path);
    }

    errno = 0;
    check(hostpid_fallback_lock_acquire_at(path, getuid() + 1U, 50) == -1 &&
              errno == EACCES,
          "unexpected owner is rejected");

#ifdef __linux__
    check(mkdtemp(ancestor_directory) != NULL,
          "symlink ancestor fixture directory");
    check(join_path(ancestor_target, sizeof(ancestor_target),
                    ancestor_directory, "/target") == 0,
          "symlink ancestor target path");
    check(join_path(ancestor_link, sizeof(ancestor_link),
                    ancestor_directory, "/link") == 0,
          "symlink ancestor link path");
    check(join_path(ancestor_nested_target,
                    sizeof(ancestor_nested_target), ancestor_target,
                    "/nested") == 0,
          "symlink ancestor nested target path");
    check(join_path(nested_path, sizeof(nested_path), ancestor_link,
                    "/nested") == 0,
          "symlink ancestor nested path");
    check(mkdir(ancestor_target, 0700) == 0,
          "symlink ancestor target directory");
    check(mkdir(ancestor_nested_target, 0700) == 0,
          "symlink ancestor nested directory");
    check(symlink(ancestor_target, ancestor_link) == 0,
          "symlink ancestor fixture");
    errno = 0;
    check(hostpid_fallback_lock_acquire_at(nested_path, getuid(), 50) == -1,
          "symlink ancestor is rejected");
    unlink(ancestor_link);
    rmdir(ancestor_nested_target);
    rmdir(ancestor_target);
    rmdir(ancestor_directory);

    snprintf(nested_path, sizeof(nested_path), "/tmp/../tmp/%s",
             strrchr(path, '/') + 1);
    errno = 0;
    check(hostpid_fallback_lock_acquire_at(nested_path, getuid(), 50) == -1 &&
              errno == EINVAL,
          "parent traversal is rejected");
#else
    (void)ancestor_directory;
    (void)ancestor_link;
    (void)ancestor_nested_target;
    (void)ancestor_target;
    (void)nested_path;
    puts("ancestor path tests skipped: Linux required");
#endif
}

static void test_live_holder_timeout(const char *path) {
    pid_t child;

    check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
          "timeout holder acquire");
    child = fork();
    check(child >= 0, "timeout contender fork");
    if (child == 0) {
        _exit(child_timeout(path, 80));
    }
    if (child > 0) {
        check(wait_for_child(child, 0) == 0, "live holder times out waiter");
    }
    check(hostpid_fallback_lock_release() == 0, "timeout holder release");
}

static void test_deadline_not_renewed(const char *path) {
    int begin_release[2];
    int ready[2];
    pid_t holder;
    pid_t waiter;
    char byte;

    if (access("/proc/self/fd", R_OK) != 0) {
        puts("deadline renewal test skipped: /proc/self/fd unavailable");
        return;
    }
    if (pipe(ready) != 0) {
        check(0, "deadline holder pipes");
        return;
    }
    if (pipe(begin_release) != 0) {
        check(0, "deadline holder pipes");
        close(ready[0]);
        close(ready[1]);
        return;
    }
    holder = fork();
    check(holder >= 0, "deadline holder fork");
    if (holder < 0) {
        close(ready[0]);
        close(ready[1]);
        close(begin_release[0]);
        close(begin_release[1]);
        return;
    }
    if (holder == 0) {
        struct timespec hold = {.tv_sec = 0, .tv_nsec = 400000000L};

        close(ready[0]);
        close(begin_release[1]);
        hostpid_fallback_lock_after_fork();
        if (hostpid_fallback_lock_acquire_at(path, getuid(), 500) != 0 ||
            write(ready[1], "1", 1) != 1) {
            _exit(2);
        }
        close(ready[1]);
        if (read(begin_release[0], &byte, 1) != 1) {
            _exit(3);
        }
        close(begin_release[0]);
        while (nanosleep(&hold, &hold) != 0 && errno == EINTR) {
        }
        _exit(hostpid_fallback_lock_release() == 0 ? 0 : 4);
    }
    close(ready[1]);
    close(begin_release[0]);
    if (read(ready[0], &byte, 1) != 1) {
        check(0, "deadline holder acquired");
        close(ready[0]);
        close(begin_release[1]);
        kill(holder, SIGKILL);
        waitpid(holder, NULL, 0);
        return;
    }
    close(ready[0]);

    waiter = fork();
    check(waiter >= 0, "deadline waiter fork");
    if (waiter == 0) {
        _exit(child_timeout(path, 80));
    }
    if (waiter > 0) {
        check(wait_until_child_opened(waiter, path) == 0,
              "deadline waiter opened the lock object");
        check(write(begin_release[1], "1", 1) == 1,
              "deadline holder release timer started");
        close(begin_release[1]);
        check(wait_for_child(waiter, 0) == 0,
              "deadline is not renewed by owner release");
    } else {
        close(begin_release[1]);
    }
    check(wait_for_child(holder, 0) == 0,
          "deadline holder release");
}

static void test_waiter_cancellation(const char *path) {
    int started[2];
    pid_t child;
    char byte;
    int status;

    check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
          "cancellation holder acquire");
    if (pipe(started) != 0) {
        check(0, "cancellation start pipe");
        hostpid_fallback_lock_release();
        return;
    }
    child = fork();
    check(child >= 0, "cancellation waiter fork");
    if (child < 0) {
        close(started[0]);
        close(started[1]);
        hostpid_fallback_lock_release();
        return;
    }
    if (child == 0) {
        close(started[0]);
        hostpid_fallback_lock_after_fork();
        if (write(started[1], "1", 1) != 1) {
            _exit(3);
        }
        close(started[1]);
        if (hostpid_fallback_lock_acquire_at(path, getuid(), 5000) == 0) {
            hostpid_fallback_lock_release();
            _exit(4);
        }
        _exit(5);
    }
    if (child > 0) {
        close(started[1]);
        check(read(started[0], &byte, 1) == 1,
              "cancellation waiter started");
        close(started[0]);
        if (access("/proc/self/fd", R_OK) == 0) {
            check(wait_until_child_opened(child, path) == 0,
                  "cancellation waiter opened the lock object");
        }
        check(kill(child, SIGKILL) == 0,
              "cancellation waiter killed");
        check(waitpid(child, &status, 0) == child &&
                  WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL,
              "cancellation waiter reaped");
    }
    check(hostpid_fallback_lock_release() == 0,
          "cancellation holder release");
    child = fork();
    check(child >= 0, "post cancellation contender fork");
    if (child == 0) {
        _exit(child_acquire(path, 500));
    }
    if (child > 0) {
        check(wait_for_child(child, 0) == 0,
              "lock recovers after waiter cancellation");
    }
}

static void test_owner_death(const char *path) {
    int ready[2] = {-1, -1};
    pid_t child;
    char byte;

    if (pipe(ready) != 0) {
        check(0, "owner death pipe");
        return;
    }
    child = fork();
    check(child >= 0, "owner death fork");
    if (child == 0) {
        close(ready[0]);
        hostpid_fallback_lock_after_fork();
        if (hostpid_fallback_lock_acquire_at(path, getuid(), 500) != 0) {
            _exit(2);
        }
        if (write(ready[1], "1", 1) != 1) {
            _exit(3);
        }
        pause();
        _exit(4);
    }
    if (child > 0) {
        close(ready[1]);
        check(read(ready[0], &byte, 1) == 1, "owner acquired before death");
        kill(child, SIGKILL);
        check(waitpid(child, NULL, 0) == child, "dead owner reaped");
        check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
              "lock recovers after owner death");
        check(hostpid_fallback_lock_release() == 0,
              "owner death recovery release");
        close(ready[0]);
    }
}

static void test_fork_cleanup(const char *path) {
    pid_t child;

    check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
          "fork holder acquire");
    child = fork();
    check(child >= 0, "fork cleanup child");
    if (child == 0) {
        hostpid_fallback_lock_after_fork();
        _exit(hostpid_fallback_lock_active_fd() == -1 ? 0 : 2);
    }
    if (child > 0) {
        check(wait_for_child(child, 0) == 0, "child drops inherited state");
    }

    child = fork();
    check(child >= 0, "fork cleanup contender");
    if (child == 0) {
        _exit(child_timeout(path, 80));
    }
    if (child > 0) {
        check(wait_for_child(child, 0) == 0,
              "parent keeps lock after child cleanup");
    }
    check(hostpid_fallback_lock_release() == 0, "fork holder release");

    child = fork();
    check(child >= 0, "post fork contender");
    if (child == 0) {
        _exit(child_acquire(path, 500));
    }
    if (child > 0) {
        check(wait_for_child(child, 0) == 0,
              "new process acquires after release");
    }
}

static void test_exec_cleanup(const char *path, const char *program) {
    char descriptor[32];
    pid_t child;
    int fd;

    check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
          "exec holder acquire");
    fd = hostpid_fallback_lock_active_fd();
    snprintf(descriptor, sizeof(descriptor), "%d", fd);
    child = fork();
    check(child >= 0, "exec cleanup fork");
    if (child == 0) {
        execl(program, program, "--check-closed", descriptor, NULL);
        _exit(3);
    }
    if (child > 0) {
        check(wait_for_child(child, 0) == 0,
              "exec closes inherited descriptor");
    }
    check(hostpid_fallback_lock_release() == 0, "exec holder release");
}

static void test_permission_change_while_waiting(const char *path) {
    int failures_before = failures;
    int ready[2];
    pid_t child;
    char byte;

    check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
          "permission change holder acquire");
    if (pipe(ready) != 0) {
        check(0, "permission change ready pipe");
        hostpid_fallback_lock_release();
        return;
    }
    waiter_ready_fd = ready[1];
    hostpid_fallback_lock_set_before_flock_hook(
        signal_waiter_ready);
    child = fork();
    check(child >= 0, "permission change waiter fork");
    if (child == 0) {
        close(ready[0]);
        hostpid_fallback_lock_after_fork();
        if (hostpid_fallback_lock_acquire_at(path, getuid(), 1000) == 0) {
            hostpid_fallback_lock_release();
            _exit(2);
        }
        _exit(errno == EACCES ? 0 : 3);
    }
    if (child > 0) {
        close(ready[1]);
        check(read(ready[0], &byte, 1) == 1,
              "permission change waiter passed initial validation");
        close(ready[0]);
        hostpid_fallback_lock_set_before_flock_hook(NULL);
        check(chmod(path, 0777) == 0,
              "permission change makes lock object untrusted");
        check(hostpid_fallback_lock_release() == 0,
              "permission change holder release");
        check(wait_for_child(child, 0) == 0,
              "permission change is rejected after lock acquisition");
        check(chmod(path, 0700) == 0,
              "permission change restores lock object");
    } else {
        close(ready[0]);
        close(ready[1]);
        hostpid_fallback_lock_set_before_flock_hook(NULL);
        hostpid_fallback_lock_release();
    }
    if (failures == failures_before) {
        puts("permission_change_rejected=passed");
    }
}

static void test_path_replacement(const char *path) {
    char old_path[4096];
    int failures_before = failures;
    int reset_ready[2];
    pid_t child;
    char byte;

    snprintf(old_path, sizeof(old_path), "%s.old", path);
    check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
          "replacement holder acquire");
    if (pipe(reset_ready) != 0) {
        check(0, "replacement reset pipe");
        hostpid_fallback_lock_release();
        return;
    }
    waiter_ready_fd = reset_ready[1];
    hostpid_fallback_lock_set_before_flock_hook(signal_waiter_ready);
    child = fork();
    check(child >= 0, "replacement contender fork");
    if (child == 0) {
        close(reset_ready[0]);
        hostpid_fallback_lock_after_fork();
        if (hostpid_fallback_lock_acquire_at(path, getuid(), 1000) == 0) {
            hostpid_fallback_lock_release();
            _exit(2);
        }
        _exit(errno == ESTALE ? 0 : 3);
    }
    if (child > 0) {
        close(reset_ready[1]);
        check(read(reset_ready[0], &byte, 1) == 1,
              "replacement waiter passed initial validation");
        close(reset_ready[0]);
        hostpid_fallback_lock_set_before_flock_hook(NULL);
        check(rename(path, old_path) == 0, "replace original path");
        check(mkdir(path, 0700) == 0, "create replacement path");
        check(hostpid_fallback_lock_release() == 0,
              "release replaced original object");
        check(wait_for_child(child, 0) == 0,
              "replacement is detected after acquisition");
        rmdir(path);
        check(rename(old_path, path) == 0, "restore original path");
    } else {
        close(reset_ready[0]);
        close(reset_ready[1]);
        hostpid_fallback_lock_set_before_flock_hook(NULL);
        hostpid_fallback_lock_release();
    }
    if (failures == failures_before) {
        puts("path_replacement_rejected=passed");
    }
}

static void test_ancestor_change_while_waiting(void) {
#ifdef __linux__
    char fixture[] = "/tmp/hami-hostpid-ancestor-change.XXXXXX";
    char lock_path[PATH_MAX];
    char original_parent[PATH_MAX];
    char parent[PATH_MAX];
    int failures_before = failures;
    int ready[2];
    pid_t child;
    char byte;
    char *fixture_root;

    fixture_root = mkdtemp(fixture);
    check(fixture_root != NULL, "ancestor change fixture root");
    if (fixture_root == NULL) {
        return;
    }
    check(join_path(parent, sizeof(parent), fixture_root, "/parent") == 0,
          "ancestor change parent path");
    check(join_path(original_parent, sizeof(original_parent), fixture_root,
                    "/parent-original") == 0,
          "ancestor change original parent path");
    check(join_path(lock_path, sizeof(lock_path), parent, "/lock") == 0,
          "ancestor change lock path");
    check(mkdir(parent, 0700) == 0, "ancestor change parent");
    check(mkdir(lock_path, 0700) == 0, "ancestor change lock object");
    check(hostpid_fallback_lock_acquire_at(lock_path, getuid(), 500) == 0,
          "ancestor change holder acquire");
    if (pipe(ready) != 0) {
        check(0, "ancestor change ready pipe");
        hostpid_fallback_lock_release();
        rmdir(lock_path);
        rmdir(parent);
        rmdir(fixture_root);
        return;
    }
    waiter_ready_fd = ready[1];
    hostpid_fallback_lock_set_before_flock_hook(signal_waiter_ready);
    child = fork();
    check(child >= 0, "ancestor change waiter fork");
    if (child == 0) {
        close(ready[0]);
        hostpid_fallback_lock_after_fork();
        if (hostpid_fallback_lock_acquire_at(lock_path, getuid(), 1000) ==
            0) {
            hostpid_fallback_lock_release();
            _exit(2);
        }
        _exit(errno == ELOOP || errno == ENOTDIR ? 0 : 3);
    }
    if (child > 0) {
        close(ready[1]);
        check(read(ready[0], &byte, 1) == 1,
              "ancestor change waiter passed initial validation");
        close(ready[0]);
        hostpid_fallback_lock_set_before_flock_hook(NULL);
        check(rename(parent, original_parent) == 0,
              "ancestor change moves trusted parent");
        check(symlink(original_parent, parent) == 0,
              "ancestor change installs symlink parent");
        check(hostpid_fallback_lock_release() == 0,
              "ancestor change holder release");
        check(wait_for_child(child, 0) == 0,
              "ancestor change is rejected after lock acquisition");
        check(unlink(parent) == 0, "ancestor change removes symlink parent");
        check(rename(original_parent, parent) == 0,
              "ancestor change restores trusted parent");
    } else {
        close(ready[0]);
        close(ready[1]);
        hostpid_fallback_lock_set_before_flock_hook(NULL);
        hostpid_fallback_lock_release();
    }
    rmdir(lock_path);
    rmdir(parent);
    rmdir(fixture_root);
    if (failures == failures_before) {
        puts("ancestor_change_rejected=passed");
    }
#else
    puts("ancestor change test skipped: Linux required");
#endif
}

int main(int argc, char **argv) {
    char directory[] = "/tmp/hami-hostpid-global-lock.XXXXXX";
    char *path;

    if (argc == 3 && strcmp(argv[1], "--check-closed") == 0) {
        int fd = atoi(argv[2]);

        errno = 0;
        return fcntl(fd, F_GETFD) == -1 && errno == EBADF ? 0 : 2;
    }
    if (argc == 6 && strcmp(argv[1], "--probe") == 0) {
        return probe_lock(argv[2], argv[3], argv[4], argv[5], 0);
    }
    if (argc == 6 && strcmp(argv[1], "--expect-timeout") == 0) {
        return probe_lock(argv[2], argv[3], argv[4], argv[5], 1);
    }
    if (argc == 3 && strcmp(argv[1], "--probe-default") == 0) {
        return probe_default_lock(argv[2]);
    }

    path = mkdtemp(directory);
    if (path == NULL) {
        perror("mkdtemp");
        return 1;
    }

    check(strcmp(HOSTPID_FALLBACK_LOCK_PATH,
                 "/tmp/vgpulock/hostpid") == 0,
          "default lock uses the trusted broker mount");

    test_discovery_path_selection();
    test_absolute_deadline_api(path);
    test_basic_contract(path);
    test_path_trust(path);
    test_live_holder_timeout(path);
    test_deadline_not_renewed(path);
    test_waiter_cancellation(path);
    test_owner_death(path);
    test_fork_cleanup(path);
    test_exec_cleanup(path, argv[0]);
    test_permission_change_while_waiting(path);
    test_path_replacement(path);
    test_ancestor_change_while_waiting();

    if (hostpid_fallback_lock_active_fd() >= 0) {
        hostpid_fallback_lock_release();
    }
    if (rmdir(path) != 0) {
        perror("rmdir");
        failures++;
    }
    if (failures != 0) {
        fprintf(stderr, "%d host PID fallback lock test(s) failed\n",
                failures);
        return 1;
    }
    puts("host PID fallback lock tests passed");
    return 0;
}

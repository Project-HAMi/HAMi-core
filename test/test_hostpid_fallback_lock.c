/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#include "../src/include/hostpid_fallback_lock.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
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

static int failures;

static int parse_unsigned(const char *value, unsigned long *parsed) {
    char *end = NULL;

    errno = 0;
    *parsed = strtoul(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' ? 0 : -1;
}

static int probe_lock(const char *path, const char *owner_value,
                      const char *timeout_value, const char *hold_value,
                      int expect_timeout) {
    struct timespec hold;
    unsigned long owner;
    unsigned long timeout_ms;
    unsigned long hold_ms;

    if (parse_unsigned(owner_value, &owner) != 0 ||
        parse_unsigned(timeout_value, &timeout_ms) != 0 ||
        parse_unsigned(hold_value, &hold_ms) != 0 ||
        owner > (unsigned long)(uid_t)-1 || timeout_ms > UINT_MAX ||
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
    hold.tv_sec = (time_t)(hold_ms / 1000UL);
    hold.tv_nsec = (long)(hold_ms % 1000UL) * 1000000L;
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
    unsigned long hold_ms;

    if (parse_unsigned(hold_value, &hold_ms) != 0 || hold_ms > UINT_MAX) {
        return 2;
    }
    if (hostpid_fallback_lock_acquire() != 0) {
        perror("hostpid_fallback_lock_acquire");
        return 3;
    }
    puts("acquired");
    fflush(stdout);
    hold.tv_sec = (time_t)(hold_ms / 1000UL);
    hold.tv_nsec = (long)(hold_ms % 1000UL) * 1000000L;
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

static int child_timeout(const char *path, unsigned int timeout_ms) {
    hostpid_fallback_lock_after_fork();
    if (hostpid_fallback_lock_acquire_at(path, getuid(), timeout_ms) == 0) {
        hostpid_fallback_lock_release();
        return 2;
    }
    return errno == ETIMEDOUT ? 0 : 3;
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

    snprintf(fd_directory, sizeof(fd_directory), "/proc/%ld/fd",
             (long)child);
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
    int file_fd;

    errno = 0;
    check(hostpid_fallback_lock_acquire_at("relative", getuid(), 50) == -1 &&
              errno == EINVAL,
          "relative path is rejected");

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

static void test_owner_death(const char *path) {
    int ready[2];
    pid_t child;
    char byte;

    check(pipe(ready) == 0, "owner death pipe");
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

static void test_path_replacement(const char *path) {
    char old_path[4096];
    int reset_ready[2];
    pid_t child;
    char byte;

    if (access("/proc/self/fd", R_OK) != 0) {
        puts("path replacement test skipped: /proc/self/fd unavailable");
        return;
    }
    snprintf(old_path, sizeof(old_path), "%s.old", path);
    check(pipe(reset_ready) == 0, "replacement reset pipe");
    check(hostpid_fallback_lock_acquire_at(path, getuid(), 500) == 0,
          "replacement holder acquire");
    child = fork();
    check(child >= 0, "replacement contender fork");
    if (child == 0) {
        close(reset_ready[0]);
        hostpid_fallback_lock_after_fork();
        if (write(reset_ready[1], "1", 1) != 1) {
            _exit(4);
        }
        close(reset_ready[1]);
        if (hostpid_fallback_lock_acquire_at(path, getuid(), 1000) == 0) {
            hostpid_fallback_lock_release();
            _exit(2);
        }
        _exit(errno == ESTALE ? 0 : 3);
    }
    if (child > 0) {
        close(reset_ready[1]);
        check(read(reset_ready[0], &byte, 1) == 1,
              "replacement child reset inherited state");
        close(reset_ready[0]);
        check(wait_until_child_opened(child, path) == 0,
              "replacement contender opened original object");
        check(rename(path, old_path) == 0, "replace original path");
        check(mkdir(path, 0700) == 0, "create replacement path");
        check(hostpid_fallback_lock_release() == 0,
              "release replaced original object");
        check(wait_for_child(child, 0) == 0,
              "replacement is detected after acquisition");
        rmdir(path);
        check(rename(old_path, path) == 0, "restore original path");
    }
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

    test_basic_contract(path);
    test_path_trust(path);
    test_live_holder_timeout(path);
    test_owner_death(path);
    test_fork_cleanup(path);
    test_exec_cleanup(path, argv[0]);
    test_path_replacement(path);

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

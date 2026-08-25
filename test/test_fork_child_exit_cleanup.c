/*
 * GPU-free regression for inherited atexit(exit_handler) after a child is
 * created without pthread_atfork.  glibc fork() runs the atfork reset, but
 * clone(SIGCHLD) does not.  After that child exit(0)s, the parent must still
 * own the shared-region lock and keep its process slot.
 */
#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define TEST_TIMEOUT_MS 5000.0

typedef struct {
    _Atomic int probe_entered;
} test_state_t;

static test_state_t *state;
static shared_region_t *region;

static double now_ms(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void sleep_ms(int milliseconds) {
    struct timespec ts;

    ts.tv_sec = milliseconds / 1000;
    ts.tv_nsec = (milliseconds % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
}

static int wait_for_counter(_Atomic int *counter, int expected,
                            double timeout_ms) {
    double deadline = now_ms() + timeout_ms;

    while (atomic_load_explicit(counter, memory_order_acquire) < expected) {
        if (now_ms() >= deadline) {
            return -1;
        }
        sleep_ms(1);
    }
    return 0;
}

static int wait_child_bounded(pid_t child, double timeout_ms, int *status_out) {
    double deadline = now_ms() + timeout_ms;
    int status;
    pid_t waited;

    for (;;) {
        do {
            waited = waitpid(child, &status, WNOHANG);
        } while (waited < 0 && errno == EINTR);
        if (waited == child) {
            if (status_out != NULL) {
                *status_out = status;
            }
            return 0;
        }
        if (waited < 0) {
            return -1;
        }
        if (now_ms() >= deadline) {
            return -1;
        }
        sleep_ms(1);
    }
}

static void kill_and_reap(pid_t child) {
    int status;

    if (child <= 0) {
        return;
    }
    if (waitpid(child, &status, WNOHANG) == child) {
        return;
    }
    kill(child, SIGKILL);
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
}

static pid_t clone_without_atfork(void) {
    return (pid_t)syscall(SYS_clone, SIGCHLD, 0);
}

static int map_shared_region(const char *cache_path) {
    int fd = open(cache_path, O_RDWR);

    if (fd < 0) {
        perror("open(shared-region cache)");
        return -1;
    }
    region = mmap(NULL, SHARED_REGION_SIZE_MAGIC, PROT_READ | PROT_WRITE,
                  MAP_SHARED, fd, 0);
    close(fd);
    if (region == MAP_FAILED) {
        perror("mmap(shared-region cache)");
        region = NULL;
        return -1;
    }
    return 0;
}

static int parent_slot_live(int32_t parent_pid) {
    int proc_num = atomic_load_explicit(&region->proc_num, memory_order_acquire);
    int i;

    if (proc_num < 1 || proc_num > SHARED_REGION_MAX_PROCESS_NUM) {
        return 0;
    }
    for (i = 0; i < proc_num; i++) {
        if (atomic_load_explicit(&region->procs[i].pid, memory_order_acquire) ==
            parent_pid) {
            return 1;
        }
    }
    return 0;
}

static int expect_parent_still_holds_lock(int32_t parent_pid, const char *when) {
    int sem_value = -1;
    size_t owner = atomic_load_explicit(&region->owner_pid, memory_order_acquire);

    if (sem_getvalue(&region->sem, &sem_value) != 0) {
        fprintf(stderr, "%s: sem_getvalue failed: %s\n", when, strerror(errno));
        return -1;
    }
    if (sem_value != 0) {
        fprintf(stderr, "%s: child posted parent's lock (sem=%d)\n", when,
                sem_value);
        return -1;
    }
    if (owner != (size_t)parent_pid) {
        fprintf(stderr, "%s: owner_pid=%zu, want parent %d\n", when, owner,
                parent_pid);
        return -1;
    }
    if (!parent_slot_live(parent_pid)) {
        fprintf(stderr, "%s: parent slot pid was cleared\n", when);
        return -1;
    }
    return 0;
}

static int test_clone_child_exit_does_not_drop_parent_lock(void) {
    pid_t clone_child;
    pid_t probe = -1;
    int32_t parent_pid = getpid();
    int status;
    int sem_value = -1;
    int result = -1;

    lock_shrreg();
    if (expect_parent_still_holds_lock(parent_pid, "before clone") != 0) {
        unlock_shrreg();
        return -1;
    }

    clone_child = clone_without_atfork();
    if (clone_child < 0) {
        fprintf(stderr, "clone(SIGCHLD) failed: %s\n", strerror(errno));
        unlock_shrreg();
        return -1;
    }
    if (clone_child == 0) {
        /* Inherited atexit must not impersonate the parent. */
        exit(0);
    }
    if (wait_child_bounded(clone_child, TEST_TIMEOUT_MS, &status) != 0 ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "clone child did not exit(0)\n");
        unlock_shrreg();
        return -1;
    }

    /* Direct invariants: no sleep.  The bug posts the sem and zeros the slot. */
    if (expect_parent_still_holds_lock(parent_pid, "after clone-child exit") !=
        0) {
        unlock_shrreg();
        return -1;
    }

    unlock_shrreg();
    if (sem_getvalue(&region->sem, &sem_value) != 0 || sem_value != 1) {
        fprintf(stderr, "parent unlock left sem=%d, want 1\n", sem_value);
        return -1;
    }

    probe = fork();
    if (probe < 0) {
        fprintf(stderr, "fork(probe) failed: %s\n", strerror(errno));
        return -1;
    }
    if (probe == 0) {
        ensure_initialized();
        atomic_fetch_add_explicit(&state->probe_entered, 1,
                                  memory_order_release);
        _exit(0);
    }
    if (wait_for_counter(&state->probe_entered, 1, TEST_TIMEOUT_MS) != 0 ||
        wait_child_bounded(probe, TEST_TIMEOUT_MS, &status) != 0 ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "probe could not initialize after parent unlocked\n");
        goto cleanup;
    }
    probe = -1;
    result = 0;

cleanup:
    kill_and_reap(probe);
    return result;
}

int main(void) {
    char cache_path[] = "/tmp/hami-fork-child-exit.XXXXXX";
    int cache_fd;

    cache_fd = mkstemp(cache_path);
    if (cache_fd < 0) {
        perror("mkstemp(shared-region cache)");
        return 1;
    }
    close(cache_fd);
    unlink(cache_path);
    if (setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, cache_path, 1) != 0 ||
        setenv("LIBCUDA_LOG_LEVEL", "0", 1) != 0) {
        perror("setenv");
        return 1;
    }

    state = mmap(NULL, sizeof(*state), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (state == MAP_FAILED) {
        perror("mmap(test state)");
        return 1;
    }
    memset(state, 0, sizeof(*state));
    log_utils_init();
    ensure_initialized();
    if (map_shared_region(cache_path) != 0) {
        unlink(cache_path);
        munmap(state, sizeof(*state));
        return 1;
    }

    if (test_clone_child_exit_does_not_drop_parent_lock() != 0) {
        unlink(cache_path);
        munmap(region, SHARED_REGION_SIZE_MAGIC);
        munmap(state, sizeof(*state));
        return 1;
    }

    unlink(cache_path);
    munmap(region, SHARED_REGION_SIZE_MAGIC);
    munmap(state, sizeof(*state));
    puts("fork-child exit cleanup tests passed");
    return 0;
}

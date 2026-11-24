/*
 * GPU-free regression for issue #432: memory charged by a process that has
 * exited must not be reported after the exit.  The slot outlives the
 * process until a new process registers, so a long-lived reader polling
 * nvmlDeviceGetMemoryInfo or cuMemGetInfo saw the old usage indefinitely.
 *
 * The other direction matters as much: a process whose memory the driver
 * still holds must keep counting.  That covers a process that has already
 * run exit_handler from atexit, and a process whose main thread has exited
 * (the leader shows Z) while another thread still runs.  A process past
 * exit_handler must also not block in ENSURE_RUNNING's status wait.
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define TEST_TIMEOUT_MS 5000.0
#define CHARGE ((size_t)1 << 30)
#define DEV 0

typedef enum {
    END_EXIT,
    END_UNDERSCORE_EXIT,
    END_WAIT_FOR_KILL,
    END_EXIT_HANDLER_THEN_WAIT,
    END_MAIN_THREAD_ONLY,
} child_end_t;

typedef struct {
    _Atomic int charged;
    _Atomic int runs_after_exit_handler;
} test_state_t;

static test_state_t *state;
static shared_region_t *region;
static int failures;

void exit_handler(void);

unsigned int cuda_to_nvml_map(unsigned int cudadev) {
    return cudadev;
}

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

static int wait_charged(double timeout_ms) {
    double deadline = now_ms() + timeout_ms;

    while (atomic_load_explicit(&state->charged, memory_order_acquire) == 0) {
        if (now_ms() >= deadline) {
            return -1;
        }
        sleep_ms(1);
    }
    return 0;
}

static int reap(pid_t child) {
    int status;

    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            return -1;
        }
    }
    return 0;
}

/* Read the leader's state and the process's thread count from /proc. */
static int read_stat(pid_t pid, char *state_out, int64_t *threads_out) {
    char path[64];
    char buf[512];
    FILE *fp;
    size_t n;
    char *comm_end;

    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    comm_end = strrchr(buf, ')');
    if (comm_end == NULL ||
        sscanf(comm_end + 1,
               " %c %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s"
               " %*s %*s %" SCNd64, state_out, threads_out) != 2) {
        return -1;
    }
    return 0;
}

/* Wait until the leader is a zombie with exactly want_threads threads. */
static int wait_zombie(pid_t child, int64_t want_threads, double timeout_ms) {
    double deadline = now_ms() + timeout_ms;

    for (;;) {
        char state_char;
        int64_t threads;

        if (read_stat(child, &state_char, &threads) == 0 &&
            state_char == 'Z' && threads == want_threads) {
            return 0;
        }
        if (now_ms() >= deadline) {
            return -1;
        }
        sleep_ms(1);
    }
}

static int proc_num(void) {
    return atomic_load_explicit(&region->proc_num, memory_order_acquire);
}

static shrreg_proc_slot_t *find_slot(int32_t pid) {
    for (int i = 0; i < proc_num(); i++) {
        if (atomic_load_explicit(&region->procs[i].pid, memory_order_acquire) ==
            pid) {
            return &region->procs[i];
        }
    }
    return NULL;
}

static uint64_t report(void) {
    return get_current_device_memory_usage(DEV);
}

static void expect_u64(const char *what, uint64_t got, uint64_t want) {
    if (got != want) {
        fprintf(stderr, "%s: got %" PRIu64 ", want %" PRIu64 "\n", what, got,
                want);
        failures++;
    }
}

static void expect_true(const char *what, int condition) {
    if (!condition) {
        fprintf(stderr, "%s\n", what);
        failures++;
    }
}

static void *block_forever(void *arg) {
    (void)arg;
    for (;;) {
        pause();
    }
    return NULL;
}

static pid_t spawn_charged_child(child_end_t end) {
    pid_t child;

    atomic_store_explicit(&state->charged, 0, memory_order_release);
    child = fork();
    if (child < 0) {
        perror("fork");
        return -1;
    }
    if (child == 0) {
        pthread_t thread;

        ensure_initialized();
        if (add_gpu_device_memory_usage(getpid(), DEV, CHARGE, 2) != 0) {
            _exit(3);
        }
        if (end == END_EXIT_HANDLER_THEN_WAIT) {
            /* What atexit runs before the driver tears the context down. */
            exit_handler();
            /* ENSURE_RUNNING spins on this; teardown calls must not block. */
            atomic_store_explicit(&state->runs_after_exit_handler,
                                  wait_status_self(1), memory_order_release);
        }
        if (end == END_MAIN_THREAD_ONLY &&
            pthread_create(&thread, NULL, block_forever, NULL) != 0) {
            _exit(4);
        }
        atomic_store_explicit(&state->charged, 1, memory_order_release);
        switch (end) {
        case END_EXIT:
            exit(0);
        case END_UNDERSCORE_EXIT:
            _exit(0);
        case END_MAIN_THREAD_ONLY:
            /* End the leader thread only; the other thread keeps running. */
            syscall(SYS_exit, 0);
            break;
        case END_WAIT_FOR_KILL:
        case END_EXIT_HANDLER_THEN_WAIT:
            break;
        }
        for (;;) {
            pause();
        }
    }
    if (wait_charged(TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "child %d never charged its slot\n", child);
        kill(child, SIGKILL);
        reap(child);
        return -1;
    }
    return child;
}

static void expect_gone(const char *name, pid_t child) {
    char what[128];

    snprintf(what, sizeof(what), "%s: report after exit", name);
    expect_u64(what, report(), 0);
    snprintf(what, sizeof(what), "%s: exited slot left in the table", name);
    expect_true(what, find_slot(child) == NULL && proc_num() == 1);
}

/* A live process keeps counting, and no reader removes its slot. */
static void test_live_child_counts(void) {
    pid_t child = spawn_charged_child(END_WAIT_FOR_KILL);

    if (child < 0) {
        failures++;
        return;
    }
    expect_u64("live: report", report(), CHARGE);
    expect_u64("live: oom view", get_gpu_memory_usage(DEV), CHARGE);
    expect_true("live: slot was removed", find_slot(child) != NULL);
    kill(child, SIGKILL);
    reap(child);
    expect_gone("live then SIGKILL", child);
}

/*
 * exit_handler runs from atexit, while the process still holds its memory
 * and its teardown still frees through the slot.  It must keep counting.
 */
static void test_exit_handler_keeps_counting_until_gone(void) {
    pid_t child = spawn_charged_child(END_EXIT_HANDLER_THEN_WAIT);
    shrreg_proc_slot_t *slot;

    if (child < 0) {
        failures++;
        return;
    }
    slot = find_slot(child);
    expect_true("exiting: exit_handler dropped the slot's pid", slot != NULL);
    if (slot != NULL) {
        expect_true("exiting: exit_handler did not mark the slot",
                    atomic_load_explicit(&slot->status, memory_order_acquire) == 0);
    }
    expect_true("exiting: CUDA calls after exit_handler would block",
                atomic_load_explicit(&state->runs_after_exit_handler,
                                     memory_order_acquire) == 1);
    expect_u64("exiting: report", report(), CHARGE);
    expect_u64("exiting: oom view", get_gpu_memory_usage(DEV), CHARGE);
    kill(child, SIGKILL);
    reap(child);
    expect_gone("exiting then gone", child);
}

static void test_ended_child(const char *name, child_end_t end, int sigkill,
                             int leave_zombie) {
    pid_t child = spawn_charged_child(end);

    if (child < 0) {
        failures++;
        return;
    }
    if (sigkill) {
        kill(child, SIGKILL);
    }
    if (leave_zombie) {
        if (wait_zombie(child, 1, TEST_TIMEOUT_MS) != 0) {
            fprintf(stderr, "%s: child %d never became a zombie\n", name, child);
            failures++;
        }
    } else if (reap(child) != 0) {
        fprintf(stderr, "%s: waitpid failed: %s\n", name, strerror(errno));
        failures++;
    }
    expect_gone(name, child);
    if (leave_zombie) {
        reap(child);
    }
}

/*
 * The leader shows Z as soon as its own thread exits, but the process and
 * its memory live on in the other threads.  Only the last thread counts.
 */
static void test_zombie_leader_with_live_thread(void) {
    pid_t child = spawn_charged_child(END_MAIN_THREAD_ONLY);

    if (child < 0) {
        failures++;
        return;
    }
    if (wait_zombie(child, 2, TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "leader zombie: child %d never reached Z with 2 threads\n",
                child);
        failures++;
    }
    expect_u64("leader zombie: report", report(), CHARGE);
    expect_true("leader zombie: slot was removed", find_slot(child) != NULL);

    kill(child, SIGKILL);
    if (wait_zombie(child, 1, TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "leader zombie: child %d never lost its last thread\n",
                child);
        failures++;
    }
    expect_gone("leader zombie, last thread gone", child);
    reap(child);
}

/* A reader must not wait for the lock; it still reports the right value. */
static void test_reader_does_not_wait_for_lock(void) {
    pid_t child = spawn_charged_child(END_WAIT_FOR_KILL);

    if (child < 0) {
        failures++;
        return;
    }
    kill(child, SIGKILL);
    reap(child);

    lock_shrreg();
    double start = now_ms();
    uint64_t usage = report();
    expect_true("lock held: reader waited for the lock",
                now_ms() - start < 1000.0);
    expect_u64("lock held: report", usage, 0);
    expect_true("lock held: slot removed without the lock",
                find_slot(child) != NULL);
    unlock_shrreg();

    expect_gone("lock released", child);
}

int main(void) {
    char cache_path[] = "/tmp/hami-exited-slot.XXXXXX";
    int cache_fd = mkstemp(cache_path);

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
    log_utils_init();
    ensure_initialized();

    int fd = open(cache_path, O_RDWR);
    if (fd < 0) {
        perror("open(shared-region cache)");
        return 1;
    }
    region = mmap(NULL, SHARED_REGION_SIZE_MAGIC, PROT_READ | PROT_WRITE,
                  MAP_SHARED, fd, 0);
    close(fd);
    if (region == MAP_FAILED) {
        perror("mmap(shared-region cache)");
        return 1;
    }

    test_live_child_counts();
    test_exit_handler_keeps_counting_until_gone();
    test_ended_child("exit(0)", END_EXIT, 0, 0);
    test_ended_child("_exit(0)", END_UNDERSCORE_EXIT, 0, 0);
    test_ended_child("SIGKILL", END_WAIT_FOR_KILL, 1, 0);
    test_ended_child("SIGKILL zombie", END_WAIT_FOR_KILL, 1, 1);
    test_zombie_leader_with_live_thread();
    test_reader_does_not_wait_for_lock();

    unlink(cache_path);
    munmap(region, SHARED_REGION_SIZE_MAGIC);
    munmap(state, sizeof(*state));
    if (failures != 0) {
        fprintf(stderr, "%d exited slot check(s) failed\n", failures);
        return 1;
    }
    puts("exited slot usage tests passed");
    return 0;
}

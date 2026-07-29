#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define DEFAULT_WORKERS 128
#define MAX_WORKERS 256
#define DEFAULT_TIMEOUT_MS 20000.0

/*
 * No wall-time ceiling is enforced unless SHRREG_TEST_MAX_WALL_MS is set.
 * Registration latency scales with the runner's core count and load, so a
 * default ceiling would fail for reasons unrelated to the code under test.
 * The timings below are always reported; only the deadlock deadline and the
 * correctness checks decide the exit status.
 */
#define WALL_CEILING_DISABLED 0.0

typedef struct {
    pid_t pid;
    int completed;
    double start_ms;
    double acquire_begin_ms;
    double acquired_ms;
    double unlock_begin_ms;
    double released_ms;
    double end_ms;
} worker_result_t;

typedef struct {
    _Atomic int ready;
    _Atomic int done;
    worker_result_t workers[MAX_WORKERS];
} test_state_t;

static test_state_t *test_state;
static int worker_index = -1;

static double now_ms(void) {
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        return 0.0;
    }
    return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1000000.0;
}

void shrreg_test_sequence_point(int sequence) {
    worker_result_t *result;
    double timestamp;

    if (test_state == NULL || worker_index < 0) {
        return;
    }
    result = &test_state->workers[worker_index];
    timestamp = now_ms();
    switch (sequence) {
        case SEQ_BEFORE_ACQUIRE_SEMLOCK:
            result->acquire_begin_ms = timestamp;
            break;
        case SEQ_ACQUIRE_SEMLOCK_OK:
            result->acquired_ms = timestamp;
            break;
        case SEQ_BEFORE_UNLOCK_SHRREG:
            result->unlock_begin_ms = timestamp;
            break;
        case SEQ_RELEASE_SEMLOCK_OK:
            result->released_ms = timestamp;
            break;
        default:
            break;
    }
}

static int compare_double(const void *left, const void *right) {
    double lhs = *(const double *)left;
    double rhs = *(const double *)right;

    return (lhs > rhs) - (lhs < rhs);
}

static double percentile(double *values, int count, int numerator) {
    int index = (numerator * count + 99) / 100 - 1;

    if (index < 0) {
        index = 0;
    } else if (index >= count) {
        index = count - 1;
    }
    return values[index];
}

static int read_positive_env(const char *name, int fallback, int maximum) {
    const char *value = getenv(name);
    char *end = NULL;
    int64_t parsed;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    errno = 0;
    parsed = strtoll(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed <= 0 || parsed > maximum) {
        fprintf(stderr, "%s must be in 1..%d\n", name, maximum);
        return -1;
    }
    return (int)parsed;
}

static double read_positive_double_env(const char *name, double fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    double parsed;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    errno = 0;
    parsed = strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0.0) {
        fprintf(stderr, "%s must be a positive number\n", name);
        return -1.0;
    }
    return parsed;
}

static int wait_for_counter(_Atomic int *counter, int expected,
                            double deadline_ms) {
    struct timespec pause = {.tv_sec = 0, .tv_nsec = 1000000};

    while (atomic_load_explicit(counter, memory_order_acquire) < expected) {
        if (now_ms() >= deadline_ms) {
            return -1;
        }
        nanosleep(&pause, NULL);
    }
    return 0;
}

static int wait_for_pipe_close(int fd) {
    char byte;
    ssize_t bytes;

    do {
        bytes = read(fd, &byte, 1);
    } while (bytes < 0 && errno == EINTR);
    return bytes == 0 ? 0 : -1;
}

static void stop_workers(pid_t *pids, int count, int signal_number) {
    int i;

    for (i = 0; i < count; i++) {
        if (pids[i] > 0) {
            kill(pids[i], signal_number);
        }
    }
}

static int inspect_region(const char *cache_path, pid_t *pids, int count) {
    shared_region_t *region;
    int fd;
    int proc_num;
    int i;
    int j;

    fd = open(cache_path, O_RDONLY);
    if (fd < 0) {
        perror("open(shared region)");
        return -1;
    }
    region = mmap(NULL, sizeof(*region), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (region == MAP_FAILED) {
        perror("mmap(shared region)");
        return -1;
    }

    proc_num = atomic_load_explicit(&region->proc_num, memory_order_acquire);
    if (proc_num != count) {
        fprintf(stderr, "expected %d live slots, found %d\n", count, proc_num);
        munmap(region, sizeof(*region));
        return -1;
    }
    for (i = 0; i < count; i++) {
        int matches = 0;

        for (j = 0; j < proc_num; j++) {
            pid_t slot_pid = atomic_load_explicit(
                &region->procs[j].pid, memory_order_acquire);
            if (slot_pid == pids[i]) {
                matches++;
            }
        }
        if (matches != 1) {
            fprintf(stderr, "PID %d appears in %d slots\n", pids[i], matches);
            munmap(region, sizeof(*region));
            return -1;
        }
    }
    munmap(region, sizeof(*region));
    return 0;
}

static void print_results(int workers, double wall_ms) {
    double before_lock[MAX_WORKERS];
    double wait[MAX_WORKERS];
    double hold[MAX_WORKERS];
    double total[MAX_WORKERS];
    double before_lock_sum = 0.0;
    double wait_sum = 0.0;
    double hold_sum = 0.0;
    double total_sum = 0.0;
    int i;

    for (i = 0; i < workers; i++) {
        worker_result_t *result = &test_state->workers[i];

        before_lock[i] = result->acquire_begin_ms - result->start_ms;
        wait[i] = result->acquired_ms - result->acquire_begin_ms;
        hold[i] = result->unlock_begin_ms - result->acquired_ms;
        total[i] = result->end_ms - result->start_ms;
        before_lock_sum += before_lock[i];
        wait_sum += wait[i];
        hold_sum += hold[i];
        total_sum += total[i];
    }
    qsort(before_lock, workers, sizeof(double), compare_double);
    qsort(wait, workers, sizeof(double), compare_double);
    qsort(hold, workers, sizeof(double), compare_double);
    qsort(total, workers, sizeof(double), compare_double);

    printf("{\"workers\":%d,\"wall_ms\":%.3f,"
           "\"before_lock_mean_ms\":%.3f,\"before_lock_p99_ms\":%.3f,"
           "\"wait_mean_ms\":%.3f,\"wait_p99_ms\":%.3f,"
           "\"hold_mean_ms\":%.3f,\"hold_p99_ms\":%.3f,"
           "\"total_mean_ms\":%.3f,\"total_p99_ms\":%.3f}\n",
           workers, wall_ms,
           before_lock_sum / workers, percentile(before_lock, workers, 99),
           wait_sum / workers, percentile(wait, workers, 99),
           hold_sum / workers, percentile(hold, workers, 99),
           total_sum / workers, percentile(total, workers, 99));
}

int main(void) {
    char cache_path[] = "/tmp/hami-shrreg-test.XXXXXX";
    pid_t pids[MAX_WORKERS] = {0};
    int start_pipe[2] = {-1, -1};
    int hold_pipe[2] = {-1, -1};
    int workers;
    int cache_fd;
    int started = 0;
    int status = 1;
    int i;
    double timeout_ms;
    double max_wall_ms;
    double release_ms;
    double wall_ms;

    workers = read_positive_env(
        "SHRREG_TEST_WORKERS", DEFAULT_WORKERS, MAX_WORKERS);
    timeout_ms = read_positive_double_env(
        "SHRREG_TEST_TIMEOUT_MS", DEFAULT_TIMEOUT_MS);
    max_wall_ms = read_positive_double_env(
        "SHRREG_TEST_MAX_WALL_MS", WALL_CEILING_DISABLED);
    if (workers < 0 || timeout_ms < 0.0 || max_wall_ms < 0.0) {
        return 2;
    }

    cache_fd = mkstemp(cache_path);
    if (cache_fd < 0) {
        perror("mkstemp");
        return 1;
    }
    close(cache_fd);
    if (unlink(cache_path) != 0) {
        perror("unlink(initial cache)");
        return 1;
    }
    if (setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, cache_path, 1) != 0 ||
        setenv("LIBCUDA_LOG_LEVEL", "0", 1) != 0) {
        perror("setenv");
        return 1;
    }
    log_utils_init();

    test_state = mmap(NULL, sizeof(*test_state),
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (test_state == MAP_FAILED) {
        perror("mmap(test state)");
        return 1;
    }
    memset(test_state, 0, sizeof(*test_state));
    atomic_init(&test_state->ready, 0);
    atomic_init(&test_state->done, 0);
    if (pipe(start_pipe) != 0 || pipe(hold_pipe) != 0) {
        perror("pipe");
        goto cleanup;
    }

    for (i = 0; i < workers; i++) {
        pid_t child = fork();

        if (child < 0) {
            perror("fork");
            goto cleanup;
        }
        if (child == 0) {
            worker_result_t *result;

            close(start_pipe[1]);
            close(hold_pipe[1]);
            worker_index = i;
            result = &test_state->workers[i];
            result->pid = getpid();
            atomic_fetch_add_explicit(
                &test_state->ready, 1, memory_order_release);
            if (wait_for_pipe_close(start_pipe[0]) != 0) {
                _exit(2);
            }

            result->start_ms = now_ms();
            ensure_initialized();
            result->end_ms = now_ms();
            result->completed = 1;
            atomic_fetch_add_explicit(
                &test_state->done, 1, memory_order_release);

            if (wait_for_pipe_close(hold_pipe[0]) != 0) {
                _exit(3);
            }
            _exit(0);
        }
        pids[i] = child;
        started++;
    }
    close(start_pipe[0]);
    start_pipe[0] = -1;
    close(hold_pipe[0]);
    hold_pipe[0] = -1;

    if (wait_for_counter(&test_state->ready, workers,
                         now_ms() + timeout_ms) != 0) {
        fprintf(stderr, "timed out waiting for %d workers at barrier\n",
                workers);
        goto cleanup;
    }
    release_ms = now_ms();
    close(start_pipe[1]);
    start_pipe[1] = -1;

    if (wait_for_counter(&test_state->done, workers,
                         release_ms + timeout_ms) != 0) {
        fprintf(stderr, "timed out after %d/%d registrations\n",
                atomic_load_explicit(&test_state->done, memory_order_acquire),
                workers);
        goto cleanup;
    }
    wall_ms = now_ms() - release_ms;
    if (inspect_region(cache_path, pids, workers) != 0) {
        goto cleanup;
    }
    for (i = 0; i < workers; i++) {
        if (!test_state->workers[i].completed ||
            test_state->workers[i].pid != pids[i] ||
            test_state->workers[i].acquire_begin_ms <= 0.0 ||
            test_state->workers[i].acquired_ms <
                test_state->workers[i].acquire_begin_ms ||
            test_state->workers[i].unlock_begin_ms <
                test_state->workers[i].acquired_ms ||
            test_state->workers[i].released_ms <
                test_state->workers[i].unlock_begin_ms) {
            fprintf(stderr, "worker %d produced incomplete timing data\n", i);
            goto cleanup;
        }
    }

    print_results(workers, wall_ms);
    if (max_wall_ms > WALL_CEILING_DISABLED && wall_ms > max_wall_ms) {
        fprintf(stderr, "registration wall time %.3f ms exceeds %.3f ms\n",
                wall_ms, max_wall_ms);
        goto cleanup;
    }
    status = 0;

cleanup:
    if (hold_pipe[1] >= 0) {
        close(hold_pipe[1]);
        hold_pipe[1] = -1;
    }
    if (status != 0) {
        stop_workers(pids, started, SIGKILL);
    }
    for (i = 0; i < started; i++) {
        int child_status;
        pid_t waited;

        do {
            waited = waitpid(pids[i], &child_status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited < 0) {
            perror("waitpid");
            status = 1;
            continue;
        }
        if (status == 0 &&
            (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0)) {
            fprintf(stderr, "worker %d exited abnormally\n", i);
            status = 1;
        }
    }
    if (start_pipe[0] >= 0) {
        close(start_pipe[0]);
    }
    if (start_pipe[1] >= 0) {
        close(start_pipe[1]);
    }
    if (hold_pipe[0] >= 0) {
        close(hold_pipe[0]);
    }
    if (hold_pipe[1] >= 0) {
        close(hold_pipe[1]);
    }
    unlink(cache_path);
    if (test_state != NULL && test_state != MAP_FAILED) {
        munmap(test_state, sizeof(*test_state));
    }
    return status;
}

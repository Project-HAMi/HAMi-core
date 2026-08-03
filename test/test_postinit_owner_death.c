/*
 * GPU-free regression tests for the postInit record lock.
 *
 * The process test kills both an initial owner and the first waiter that
 * acquires after it.  A second waiter must remain blocked while either owner
 * is alive and must acquire promptly after each SIGKILL.  The thread test
 * verifies the process-local guard required by POSIX record-lock semantics.
 */
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define PROCESS_COUNT 3
#define HOLDER_ID 0
#define FIRST_WAITER_ID 1
#define SECOND_WAITER_ID 2
#define TEST_TIMEOUT_MS 5000.0
#define OBSERVATION_MS 200
#define THREAD_COUNT 2
#define THREAD_HOLD_MS 100

typedef struct {
    _Atomic int process_started;
    _Atomic int process_acquired;
    _Atomic int process_completed;
    _Atomic int process_failures;
    _Atomic int current_owner;
    _Atomic int release_process[PROCESS_COUNT];

    _Atomic int threads_started;
    _Atomic int threads_go;
    _Atomic int threads_acquired;
    _Atomic int threads_completed;
    _Atomic int threads_active;
    _Atomic int max_threads_active;
    _Atomic int thread_failures;
} test_state_t;

static test_state_t *state;

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
    pid_t waited;

    if (child <= 0) {
        return;
    }
    do {
        waited = waitpid(child, &status, WNOHANG);
    } while (waited < 0 && errno == EINTR);
    if (waited == child || (waited < 0 && errno == ECHILD)) {
        return;
    }
    kill(child, SIGKILL);
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
}

static int kill_and_expect_sigkill(pid_t child) {
    int status;

    if (kill(child, SIGKILL) != 0 ||
        wait_child_bounded(child, TEST_TIMEOUT_MS, &status) != 0) {
        return -1;
    }
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL ? 0 : -1;
}

static void initialize_state(void) {
    int i;

    atomic_init(&state->process_started, 0);
    atomic_init(&state->process_acquired, 0);
    atomic_init(&state->process_completed, 0);
    atomic_init(&state->process_failures, 0);
    atomic_init(&state->current_owner, -1);
    for (i = 0; i < PROCESS_COUNT; i++) {
        atomic_init(&state->release_process[i], 0);
    }
    atomic_init(&state->threads_started, 0);
    atomic_init(&state->threads_go, 0);
    atomic_init(&state->threads_acquired, 0);
    atomic_init(&state->threads_completed, 0);
    atomic_init(&state->threads_active, 0);
    atomic_init(&state->max_threads_active, 0);
    atomic_init(&state->thread_failures, 0);
}

static void process_worker(int id) {
    ensure_initialized();
    atomic_fetch_add_explicit(&state->process_started, 1,
                              memory_order_release);
    if (lock_postinit() != 1) {
        atomic_fetch_add_explicit(&state->process_failures, 1,
                                  memory_order_release);
        _exit(2);
    }

    atomic_store_explicit(&state->current_owner, id, memory_order_release);
    atomic_fetch_add_explicit(&state->process_acquired, 1,
                              memory_order_release);
    while (atomic_load_explicit(&state->release_process[id],
                                memory_order_acquire) == 0) {
        sleep_ms(1);
    }
    unlock_postinit();
    atomic_fetch_add_explicit(&state->process_completed, 1,
                              memory_order_release);
    _exit(0);
}

static pid_t spawn_process_worker(int id) {
    pid_t child = fork();

    if (child == 0) {
        process_worker(id);
    }
    return child;
}

static int test_process_owner_death(void) {
    pid_t children[PROCESS_COUNT] = {0};
    int first_recovery;
    int survivor;
    int status;
    int result = -1;

    children[HOLDER_ID] = spawn_process_worker(HOLDER_ID);
    if (children[HOLDER_ID] < 0 ||
        wait_for_counter(&state->process_acquired, 1, TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "initial postinit owner did not acquire\n");
        goto cleanup;
    }

    children[FIRST_WAITER_ID] = spawn_process_worker(FIRST_WAITER_ID);
    children[SECOND_WAITER_ID] = spawn_process_worker(SECOND_WAITER_ID);
    if (children[FIRST_WAITER_ID] < 0 || children[SECOND_WAITER_ID] < 0 ||
        wait_for_counter(&state->process_started, PROCESS_COUNT,
                         TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "postinit waiters did not start\n");
        goto cleanup;
    }

    sleep_ms(OBSERVATION_MS);
    if (atomic_load_explicit(&state->process_acquired,
                             memory_order_acquire) != 1) {
        fprintf(stderr, "a waiter overlapped the live initial owner\n");
        goto cleanup;
    }

    if (kill_and_expect_sigkill(children[HOLDER_ID]) != 0) {
        fprintf(stderr, "could not SIGKILL the initial owner\n");
        goto cleanup;
    }
    children[HOLDER_ID] = 0;
    if (wait_for_counter(&state->process_acquired, 2, TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "no waiter recovered after initial owner death\n");
        goto cleanup;
    }
    first_recovery = atomic_load_explicit(&state->current_owner,
                                          memory_order_acquire);
    if (first_recovery != FIRST_WAITER_ID &&
        first_recovery != SECOND_WAITER_ID) {
        fprintf(stderr, "recovery owner id was invalid\n");
        goto cleanup;
    }

    sleep_ms(OBSERVATION_MS);
    if (atomic_load_explicit(&state->process_acquired,
                             memory_order_acquire) != 2) {
        fprintf(stderr, "both waiters entered the critical section\n");
        goto cleanup;
    }

    if (kill_and_expect_sigkill(children[first_recovery]) != 0) {
        fprintf(stderr, "could not SIGKILL the first recovery owner\n");
        goto cleanup;
    }
    children[first_recovery] = 0;
    if (wait_for_counter(&state->process_acquired, 3, TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "second waiter did not recover after repeated owner death\n");
        goto cleanup;
    }
    survivor = atomic_load_explicit(&state->current_owner,
                                    memory_order_acquire);
    if ((survivor != FIRST_WAITER_ID && survivor != SECOND_WAITER_ID) ||
        survivor == first_recovery) {
        fprintf(stderr, "surviving recovery owner id was invalid\n");
        goto cleanup;
    }

    atomic_store_explicit(&state->release_process[survivor], 1,
                          memory_order_release);
    if (wait_for_counter(&state->process_completed, 1, TEST_TIMEOUT_MS) != 0 ||
        wait_child_bounded(children[survivor], TEST_TIMEOUT_MS, &status) != 0 ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "surviving recovery waiter did not unlock cleanly\n");
        goto cleanup;
    }
    children[survivor] = 0;
    if (atomic_load_explicit(&state->process_failures,
                             memory_order_acquire) != 0) {
        fprintf(stderr, "a process failed to acquire the postinit lock\n");
        goto cleanup;
    }

    if (lock_postinit() != 1) {
        fprintf(stderr, "postinit lock was not reusable after recovery\n");
        goto cleanup;
    }
    unlock_postinit();
    result = 0;

cleanup:
    for (int i = 0; i < PROCESS_COUNT; i++) {
        kill_and_reap(children[i]);
    }
    return result;
}

static void record_thread_entry(void) {
    int active = atomic_fetch_add_explicit(&state->threads_active, 1,
                                           memory_order_acq_rel) + 1;
    int observed = atomic_load_explicit(&state->max_threads_active,
                                        memory_order_acquire);

    while (active > observed &&
           !atomic_compare_exchange_weak_explicit(
               &state->max_threads_active, &observed, active,
               memory_order_release, memory_order_acquire)) {
    }
}

static void *thread_worker(void *unused) {
    (void)unused;
    atomic_fetch_add_explicit(&state->threads_started, 1,
                              memory_order_release);
    while (atomic_load_explicit(&state->threads_go, memory_order_acquire) == 0) {
        sleep_ms(1);
    }
    if (lock_postinit() != 1) {
        atomic_fetch_add_explicit(&state->thread_failures, 1,
                                  memory_order_release);
        return NULL;
    }
    record_thread_entry();
    atomic_fetch_add_explicit(&state->threads_acquired, 1,
                              memory_order_release);
    sleep_ms(THREAD_HOLD_MS);
    atomic_fetch_sub_explicit(&state->threads_active, 1,
                              memory_order_acq_rel);
    unlock_postinit();
    atomic_fetch_add_explicit(&state->threads_completed, 1,
                              memory_order_release);
    return NULL;
}

static int test_same_process_threads(void) {
    pthread_t threads[THREAD_COUNT];
    int created = 0;

    for (int i = 0; i < THREAD_COUNT; i++) {
        if (pthread_create(&threads[i], NULL, thread_worker, NULL) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return -1;
        }
        created++;
    }
    if (wait_for_counter(&state->threads_started, THREAD_COUNT,
                         TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "postinit lock threads did not start\n");
        return -1;
    }
    atomic_store_explicit(&state->threads_go, 1, memory_order_release);
    if (wait_for_counter(&state->threads_completed, THREAD_COUNT,
                         TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "postinit lock threads did not finish\n");
        return -1;
    }
    for (int i = 0; i < created; i++) {
        pthread_join(threads[i], NULL);
    }
    if (atomic_load_explicit(&state->thread_failures,
                             memory_order_acquire) != 0 ||
        atomic_load_explicit(&state->threads_acquired,
                             memory_order_acquire) != THREAD_COUNT ||
        atomic_load_explicit(&state->max_threads_active,
                             memory_order_acquire) != 1) {
        fprintf(stderr, "same-process postinit calls were not serialized\n");
        return -1;
    }
    return 0;
}

int main(void) {
    char cache_path[] = "/tmp/hami-postinit-ownerdeath.XXXXXX";
    int cache_fd;
    int failures = 0;

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
    initialize_state();
    log_utils_init();
    ensure_initialized();

    if (test_process_owner_death() != 0) {
        failures++;
    }
    if (test_same_process_threads() != 0) {
        failures++;
    }

    unlink(cache_path);
    if (failures != 0) {
        fprintf(stderr, "%d postinit record-lock test group(s) failed\n",
                failures);
        return 1;
    }
    munmap(state, sizeof(*state));
    puts("postinit owner-death tests passed");
    return 0;
}

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
#include "hostpid_fallback_lock.h"

#define PROCESS_COUNT 3
#define HOLDER_ID 0
#define FIRST_WAITER_ID 1
#define SECOND_WAITER_ID 2
#define TEST_TIMEOUT_MS 5000.0
#define OBSERVATION_MS 200
#define THREAD_COUNT 2
#define THREAD_HOLD_MS 100
#define BOUNDED_CACHE_TIMEOUT_MS 120U
#define ABBA_GLOBAL_TIMEOUT_MS 150U
#define ABBA_CACHE_TIMEOUT_MS 1000U
#define SHARED_FALLBACK_TIMEOUT_MS 600U
#define SHARED_GLOBAL_HOLD_MS 350
#define SHARED_TOTAL_LIMIT_MS 900.0

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

    _Atomic int abba_cache_acquired;
    _Atomic int abba_global_acquired;

    _Atomic int local_timeout_acquired;
    _Atomic int local_timeout_release;
    _Atomic int local_timeout_failures;

    _Atomic int shared_global_ready;
    _Atomic int shared_global_start;
    _Atomic int shared_cache_ready;
    _Atomic int shared_cache_release;
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
    atomic_init(&state->abba_cache_acquired, 0);
    atomic_init(&state->abba_global_acquired, 0);
    atomic_init(&state->local_timeout_acquired, 0);
    atomic_init(&state->local_timeout_release, 0);
    atomic_init(&state->local_timeout_failures, 0);
    atomic_init(&state->shared_global_ready, 0);
    atomic_init(&state->shared_global_start, 0);
    atomic_init(&state->shared_cache_ready, 0);
    atomic_init(&state->shared_cache_release, 0);
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

static void *local_timeout_holder(void *unused) {
    (void)unused;

    if (lock_postinit() != 1) {
        atomic_fetch_add_explicit(&state->local_timeout_failures, 1,
                                  memory_order_release);
        return NULL;
    }
    atomic_store_explicit(&state->local_timeout_acquired, 1,
                          memory_order_release);
    while (atomic_load_explicit(&state->local_timeout_release,
                                memory_order_acquire) == 0) {
        sleep_ms(1);
    }
    unlock_postinit();
    return NULL;
}

static int test_same_process_live_holder_timeout(void) {
    pthread_t holder;
    double started;
    double elapsed;
    int result = -1;

    atomic_store_explicit(&state->local_timeout_acquired, 0,
                          memory_order_release);
    atomic_store_explicit(&state->local_timeout_release, 0,
                          memory_order_release);
    atomic_store_explicit(&state->local_timeout_failures, 0,
                          memory_order_release);
    if (pthread_create(&holder, NULL, local_timeout_holder, NULL) != 0) {
        fprintf(stderr, "local timeout holder creation failed\n");
        return -1;
    }
    if (wait_for_counter(&state->local_timeout_acquired, 1,
                         TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "local timeout holder did not acquire\n");
        goto cleanup;
    }

    started = now_ms();
    errno = 0;
    if (lock_postinit_timeout(BOUNDED_CACHE_TIMEOUT_MS) != 0 ||
        errno != ETIMEDOUT) {
        fprintf(stderr, "local live holder did not time out cleanly\n");
        goto cleanup;
    }
    elapsed = now_ms() - started;
    if (elapsed < 50.0 || elapsed > 1000.0) {
        fprintf(stderr, "local timeout was outside its bound: %.3f ms\n",
                elapsed);
        goto cleanup;
    }
    result = 0;

cleanup:
    atomic_store_explicit(&state->local_timeout_release, 1,
                          memory_order_release);
    pthread_join(holder, NULL);
    if (atomic_load_explicit(&state->local_timeout_failures,
                             memory_order_acquire) != 0) {
        return -1;
    }
    if (lock_postinit_timeout(500U) != 1) {
        fprintf(stderr, "local lock did not recover after timeout\n");
        return -1;
    }
    unlock_postinit();
    return result;
}

static int test_live_cache_holder_timeout(void) {
    int ready[2] = {-1, -1};
    int release[2] = {-1, -1};
    pid_t child;
    char byte;
    int status;
    double started;
    double elapsed;

    if (pipe(ready) != 0) {
        perror("cache timeout pipes");
        return -1;
    }
    if (pipe(release) != 0) {
        perror("cache timeout pipes");
        close(ready[0]);
        close(ready[1]);
        return -1;
    }
    child = fork();
    if (child == 0) {
        close(ready[0]);
        close(release[1]);
        ensure_initialized();
        if (lock_postinit() != 1 || write(ready[1], "1", 1) != 1) {
            _exit(2);
        }
        close(ready[1]);
        if (read(release[0], &byte, 1) != 1) {
            _exit(3);
        }
        close(release[0]);
        unlock_postinit();
        _exit(0);
    }
    if (child < 0) {
        perror("cache timeout fork");
        close(ready[0]);
        close(ready[1]);
        close(release[0]);
        close(release[1]);
        return -1;
    }
    close(ready[1]);
    close(release[0]);
    if (read(ready[0], &byte, 1) != 1) {
        fprintf(stderr, "cache timeout holder did not acquire\n");
        goto fail;
    }
    close(ready[0]);
    ready[0] = -1;

    started = now_ms();
    errno = 0;
    if (lock_postinit_timeout(BOUNDED_CACHE_TIMEOUT_MS) != 0 ||
        errno != ETIMEDOUT) {
        fprintf(stderr, "live cache holder did not time out cleanly\n");
        goto fail;
    }
    elapsed = now_ms() - started;
    if (elapsed < 50.0 || elapsed > 1000.0) {
        fprintf(stderr, "cache timeout was outside its bound: %.3f ms\n",
                elapsed);
        goto fail;
    }

    if (write(release[1], "1", 1) != 1) {
        goto fail;
    }
    close(release[1]);
    release[1] = -1;
    if (wait_child_bounded(child, TEST_TIMEOUT_MS, &status) != 0 ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "cache timeout holder did not exit cleanly\n");
        goto fail;
    }
    child = 0;
    if (lock_postinit_timeout(500U) != 1) {
        fprintf(stderr, "cache lock did not recover after timeout\n");
        return -1;
    }
    unlock_postinit();
    return 0;

fail:
    if (ready[0] >= 0) {
        close(ready[0]);
    }
    if (release[1] >= 0) {
        close(release[1]);
    }
    kill_and_reap(child);
    return -1;
}

static void shared_deadline_global_holder(const char *global_directory) {
    hostpid_fallback_lock_after_fork();
    if (hostpid_fallback_lock_acquire_at(global_directory, getuid(),
                                         2000U) != 0) {
        _exit(2);
    }
    atomic_store_explicit(&state->shared_global_ready, 1,
                          memory_order_release);
    while (atomic_load_explicit(&state->shared_global_start,
                                memory_order_acquire) == 0) {
        sleep_ms(1);
    }
    sleep_ms(SHARED_GLOBAL_HOLD_MS);
    if (hostpid_fallback_lock_release() != 0) {
        _exit(3);
    }
    _exit(0);
}

static void shared_deadline_cache_holder(void) {
    ensure_initialized();
    if (lock_postinit() != 1) {
        _exit(2);
    }
    atomic_store_explicit(&state->shared_cache_ready, 1,
                          memory_order_release);
    while (atomic_load_explicit(&state->shared_cache_release,
                                memory_order_acquire) == 0) {
        sleep_ms(1);
    }
    unlock_postinit();
    _exit(0);
}

static int test_shared_fallback_deadline(void) {
    char global_directory[] = "/tmp/hami-shared-deadline.XXXXXX";
    struct timespec deadline;
    pid_t global_holder = 0;
    pid_t cache_holder = 0;
    int global_acquired = 0;
    int global_status = 0;
    int cache_status = 0;
    int result = -1;
    double started;
    double global_elapsed;
    double total_elapsed;

    if (mkdtemp(global_directory) == NULL) {
        perror("mkdtemp(shared fallback deadline)");
        return -1;
    }
    atomic_store_explicit(&state->shared_global_ready, 0,
                          memory_order_release);
    atomic_store_explicit(&state->shared_global_start, 0,
                          memory_order_release);
    atomic_store_explicit(&state->shared_cache_ready, 0,
                          memory_order_release);
    atomic_store_explicit(&state->shared_cache_release, 0,
                          memory_order_release);

    global_holder = fork();
    if (global_holder == 0) {
        shared_deadline_global_holder(global_directory);
    }
    if (global_holder < 0) {
        perror("shared deadline global holder fork");
        global_holder = 0;
        goto cleanup;
    }
    cache_holder = fork();
    if (cache_holder == 0) {
        shared_deadline_cache_holder();
    }
    if (cache_holder < 0) {
        perror("shared deadline cache holder fork");
        cache_holder = 0;
        goto cleanup;
    }
    if (wait_for_counter(&state->shared_global_ready, 1,
                         TEST_TIMEOUT_MS) != 0 ||
        wait_for_counter(&state->shared_cache_ready, 1,
                         TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "shared deadline holders did not acquire\n");
        goto cleanup;
    }
    if (hostpid_fallback_lock_deadline_after_ms(
            &deadline, SHARED_FALLBACK_TIMEOUT_MS) != 0) {
        perror("shared fallback deadline");
        goto cleanup;
    }

    started = now_ms();
    atomic_store_explicit(&state->shared_global_start, 1,
                          memory_order_release);
    if (hostpid_fallback_lock_acquire_at_until(
            global_directory, getuid(), &deadline) != 0) {
        perror("shared deadline global acquisition");
        goto cleanup;
    }
    global_acquired = 1;
    global_elapsed = now_ms() - started;
    errno = 0;
    if (lock_postinit_deadline(&deadline) != 0 || errno != ETIMEDOUT) {
        fprintf(stderr, "cache lock did not consume the global deadline\n");
        goto cleanup;
    }
    total_elapsed = now_ms() - started;
    if (global_elapsed < (double)SHARED_GLOBAL_HOLD_MS / 2.0 ||
        total_elapsed < (double)SHARED_FALLBACK_TIMEOUT_MS / 2.0 ||
        total_elapsed > SHARED_TOTAL_LIMIT_MS) {
        fprintf(stderr,
                "shared fallback deadline was outside its bound: "
                "global=%.3f total=%.3f ms\n",
                global_elapsed, total_elapsed);
        goto cleanup;
    }
    printf("shared_fallback_deadline_ms=%.3f global_wait_ms=%.3f\n",
           total_elapsed, global_elapsed);
    result = 0;

cleanup:
    if (global_acquired && hostpid_fallback_lock_release() != 0) {
        fprintf(stderr, "shared deadline global release failed\n");
        result = -1;
    }
    atomic_store_explicit(&state->shared_global_start, 1,
                          memory_order_release);
    atomic_store_explicit(&state->shared_cache_release, 1,
                          memory_order_release);
    if (global_holder > 0 &&
        (wait_child_bounded(global_holder, TEST_TIMEOUT_MS,
                            &global_status) != 0 ||
         !WIFEXITED(global_status) || WEXITSTATUS(global_status) != 0)) {
        result = -1;
    }
    if (cache_holder > 0 &&
        (wait_child_bounded(cache_holder, TEST_TIMEOUT_MS,
                            &cache_status) != 0 ||
         !WIFEXITED(cache_status) || WEXITSTATUS(cache_status) != 0)) {
        result = -1;
    }
    kill_and_reap(global_holder);
    kill_and_reap(cache_holder);
    if (rmdir(global_directory) != 0) {
        result = -1;
    }
    return result;
}

static void reverse_order_worker(const char *global_directory) {
    int global_status;
    int saved_errno;

    hostpid_fallback_lock_after_fork();
    ensure_initialized();
    if (lock_postinit() != 1) {
        _exit(2);
    }
    atomic_store_explicit(&state->abba_cache_acquired, 1,
                          memory_order_release);
    if (wait_for_counter(&state->abba_global_acquired, 1,
                         TEST_TIMEOUT_MS) != 0) {
        unlock_postinit();
        _exit(3);
    }
    errno = 0;
    global_status = hostpid_fallback_lock_acquire_at(
        global_directory, getuid(), ABBA_GLOBAL_TIMEOUT_MS);
    saved_errno = errno;
    if (global_status == 0) {
        hostpid_fallback_lock_release();
    }
    unlock_postinit();
    _exit(global_status == -1 && saved_errno == ETIMEDOUT ? 0 : 4);
}

static int test_reverse_order_abba(void) {
    char global_directory[] = "/tmp/hami-postinit-abba.XXXXXX";
    pid_t child;
    int status;
    int result = -1;

    if (mkdtemp(global_directory) == NULL) {
        perror("mkdtemp(ABBA global lock)");
        return -1;
    }
    atomic_store_explicit(&state->abba_cache_acquired, 0,
                          memory_order_release);
    atomic_store_explicit(&state->abba_global_acquired, 0,
                          memory_order_release);
    child = fork();
    if (child == 0) {
        reverse_order_worker(global_directory);
    }
    if (child < 0) {
        perror("ABBA worker fork");
        rmdir(global_directory);
        return -1;
    }
    if (wait_for_counter(&state->abba_cache_acquired, 1,
                         TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "reverse-order worker did not acquire cache lock\n");
        goto cleanup;
    }
    if (hostpid_fallback_lock_acquire_at(global_directory, getuid(),
                                         500U) != 0) {
        fprintf(stderr, "normal-order worker did not acquire global lock\n");
        goto cleanup;
    }
    atomic_store_explicit(&state->abba_global_acquired, 1,
                          memory_order_release);
    if (lock_postinit_timeout(ABBA_CACHE_TIMEOUT_MS) != 1) {
        fprintf(stderr, "normal-order worker did not recover from ABBA\n");
        hostpid_fallback_lock_release();
        goto cleanup;
    }
    unlock_postinit();
    if (hostpid_fallback_lock_release() != 0) {
        fprintf(stderr, "normal-order global release failed\n");
        goto cleanup;
    }
    if (wait_child_bounded(child, TEST_TIMEOUT_MS, &status) != 0 ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "reverse-order worker did not time out cleanly\n");
        goto cleanup;
    }
    child = 0;
    result = 0;

cleanup:
    kill_and_reap(child);
    rmdir(global_directory);
    return result;
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
    if (test_same_process_live_holder_timeout() != 0) {
        failures++;
    }
    if (test_live_cache_holder_timeout() != 0) {
        failures++;
    }
    if (test_shared_fallback_deadline() != 0) {
        failures++;
    }
    if (test_reverse_order_abba() != 0) {
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

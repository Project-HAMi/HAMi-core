/*
 * GPU-free regression test for process-slot reclamation on the join path.
 *
 * Joining the shared region used to sweep every occupied slot for liveness,
 * and that sweep reads /proc/<pid>/stat once per slot while the region lock is
 * held.  The sweep now runs only once slots are scarce, so this test pins both
 * halves of that contract: below the threshold a join must leave slots held by
 * dead processes alone, and at the threshold a join must reclaim them so the
 * table cannot grow without bound.
 *
 * The target is built with a small SHARED_REGION_SWEEP_THRESHOLD so the
 * reclaim path is reachable without spawning three quarters of the table.
 */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define TEST_TIMEOUT_MS 5000.0
/* Above this the test would have to spawn too many processes to be useful. */
#define MAX_TEST_THRESHOLD 32
#define DEAD_SLOTS_BELOW_THRESHOLD 2
/* Enough cycles to cross the threshold several times over. */
#define RECLAIM_CYCLES (SHARED_REGION_SWEEP_THRESHOLD * 3)
/* A recycled PID reads as alive and survives one sweep, so allow slack. */
#define OCCUPANCY_SLACK 2

typedef struct {
    _Atomic int joined;
} test_state_t;

static test_state_t *state;
/* Read-only view of the cache file, so the test can read proc_num without
 * taking a slot of its own or reaching into the module's statics. */
static shared_region_t *region_view;

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

static void kill_and_reap(pid_t child) {
    int status;

    if (child <= 0) {
        return;
    }
    kill(child, SIGKILL);
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
}

static int map_region_view(const char *cache_path) {
    int fd = open(cache_path, O_RDONLY);

    if (fd < 0) {
        perror("open(shared-region cache)");
        return -1;
    }
    region_view = mmap(NULL, SHARED_REGION_SIZE_MAGIC, PROT_READ, MAP_SHARED,
                       fd, 0);
    close(fd);
    if (region_view == MAP_FAILED) {
        perror("mmap(shared-region cache)");
        region_view = NULL;
        return -1;
    }
    return 0;
}

static int occupied_slots(void) {
    return atomic_load_explicit(&region_view->proc_num, memory_order_acquire);
}

/* Take a slot, announce it, then wait to be killed so the slot is left behind
 * with a PID that no longer exists -- exactly what a SIGKILL'd container does,
 * since the exit handler never runs. */
static void slot_worker(void) {
    ensure_initialized();
    atomic_fetch_add_explicit(&state->joined, 1, memory_order_release);
    for (;;) {
        sleep_ms(10);
    }
}

static pid_t spawn_joined_worker(int expected_joins) {
    pid_t child = fork();

    if (child == 0) {
        slot_worker();
        _exit(0);
    }
    if (child < 0) {
        perror("fork");
        return -1;
    }
    if (wait_for_counter(&state->joined, expected_joins, TEST_TIMEOUT_MS) != 0) {
        fprintf(stderr, "worker %d did not join the shared region\n",
                expected_joins);
        kill_and_reap(child);
        return -1;
    }
    return child;
}

/* Below the threshold a join must not pay for a liveness sweep, so slots held
 * by processes that already died stay in the table. */
static int test_dead_slots_are_kept_below_threshold(int *joins) {
    pid_t child;
    int expected;
    int observed;
    int i;

    for (i = 0; i < DEAD_SLOTS_BELOW_THRESHOLD; i++) {
        child = spawn_joined_worker(++(*joins));
        if (child < 0) {
            return -1;
        }
        kill_and_reap(child);
    }

    child = spawn_joined_worker(++(*joins));
    if (child < 0) {
        return -1;
    }
    /* This process holds a slot too, hence the +1. */
    expected = 1 + DEAD_SLOTS_BELOW_THRESHOLD + 1;
    observed = occupied_slots();
    kill_and_reap(child);

    if (observed != expected) {
        fprintf(stderr,
                "join below the sweep threshold changed the table: "
                "expected %d occupied slots, saw %d\n",
                expected, observed);
        return -1;
    }
    return 0;
}

/* Once slots are scarce a join must reclaim the dead ones, so repeated
 * join-and-die cycles cannot grow the table past the threshold. */
static int test_dead_slots_are_reclaimed_at_threshold(int *joins) {
    int previous = occupied_slots();
    int peak = previous;
    int reclaims = 0;
    pid_t child;
    int observed;
    int i;

    for (i = 0; i < RECLAIM_CYCLES; i++) {
        child = spawn_joined_worker(++(*joins));
        if (child < 0) {
            return -1;
        }
        observed = occupied_slots();
        kill_and_reap(child);

        if (observed > peak) {
            peak = observed;
        }
        if (observed < previous) {
            reclaims++;
        }
        previous = observed;
    }

    if (peak > SHARED_REGION_SWEEP_THRESHOLD + OCCUPANCY_SLACK) {
        fprintf(stderr,
                "slot table grew past the sweep threshold: peak %d, "
                "threshold %d\n",
                peak, (int)SHARED_REGION_SWEEP_THRESHOLD);
        return -1;
    }
    if (reclaims == 0) {
        fprintf(stderr, "no join ever reclaimed a dead slot in %d cycles\n",
                RECLAIM_CYCLES);
        return -1;
    }
    return 0;
}

int main(void) {
    char cache_path[] = "/tmp/hami-proc-slot-reclaim.XXXXXX";
    int cache_fd;
    int joins = 0;
    int failures = 0;

    if ((int)SHARED_REGION_SWEEP_THRESHOLD > MAX_TEST_THRESHOLD) {
        printf("skipping: sweep threshold %d needs too many processes\n",
               (int)SHARED_REGION_SWEEP_THRESHOLD);
        return 0;
    }

    cache_fd = mkstemp(cache_path);
    if (cache_fd < 0) {
        perror("mkstemp(shared-region cache)");
        return 1;
    }
    close(cache_fd);
    unlink(cache_path);
    if (setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, cache_path, 1) != 0 ||
        setenv("CUDA_DEVICE_MEMORY_LIMIT", "1024m", 1) != 0 ||
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
    atomic_init(&state->joined, 0);
    log_utils_init();
    ensure_initialized();

    if (map_region_view(cache_path) != 0) {
        unlink(cache_path);
        return 1;
    }

    if (test_dead_slots_are_kept_below_threshold(&joins) != 0) {
        failures++;
    }
    if (test_dead_slots_are_reclaimed_at_threshold(&joins) != 0) {
        failures++;
    }

    munmap(region_view, SHARED_REGION_SIZE_MAGIC);
    unlink(cache_path);
    if (failures != 0) {
        fprintf(stderr, "%d process-slot reclamation test(s) failed\n",
                failures);
        return 1;
    }
    munmap(state, sizeof(*state));
    puts("process-slot reclamation tests passed");
    return 0;
}

/*
 * GPU-free characterization tests for the memory-limit shared-region source
 * of truth (HAMi#2125).
 *
 * These tests exercise the enforcement value HAMi-core would hold for a
 * process -- get_current_device_memory_limit(0), the exact input oom_check()
 * gates every cudaMalloc on (allocator.c) -- under the process-startup shapes
 * named in the issue: the original process, a child that inherited no
 * environment, and a fresh "SSH/PAM-style" clean-environment process. No CUDA
 * or NVML entry point is called, so no GPU, driver, or CUDA runtime is needed;
 * this mirrors test_postinit_owner_death.c, which links the production
 * shared-region implementation directly.
 *
 * Each scenario runs in its own child against its own fresh cache file so that
 * "first writer to touch the region" is controlled explicitly. The region is
 * seeded exactly once (multiprocess_memory_limit.c try_create_shrreg): the
 * first process's environment decides region->limit[], and every later process
 * reads it back without correcting a mismatch.
 *
 * IMPORTANT -- these assertions encode CURRENT behavior, including the #2125
 * gap, so the test is green on today's code and documents the boundary. The
 * three assertions tagged  [FIX FLIPS THIS]  are the regressions an accepted
 * fix must invert (LEAK -> ENFORCED); when the fix lands, change their expected
 * value to LIMIT_BYTES and they become the acceptance test for the issue.
 */
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define LIMIT_ENV_KEY  "CUDA_DEVICE_MEMORY_LIMIT_0"
#define LIMIT_BYTES    ((uint64_t)1073741824ULL) /* 1 GiB */
#define LIMIT_ENV_VAL  "1073741824"
#define TEST_TIMEOUT_MS 5000.0

/* Shared slot for a child to report the limit it observed back to the parent. */
typedef struct {
    _Atomic uint64_t reported;
    _Atomic int      done;
} probe_result_t;

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

static int wait_child_bounded(pid_t child, double timeout_ms) {
    double deadline = now_ms() + timeout_ms;
    int status;
    pid_t waited;

    for (;;) {
        do {
            waited = waitpid(child, &status, WNOHANG);
        } while (waited < 0 && errno == EINTR);
        if (waited == child) {
            return (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 0 : -1;
        }
        if (now_ms() >= deadline) {
            /* The child is still running and holds the shared region and cache
             * file; kill and reap it so a later probe cannot observe its state. */
            kill(child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
            }
            return -1;
        }
        if (waited < 0) {
            return -1;
        }
        sleep_ms(1);
    }
}

/*
 * Run one probe in a fresh child: set up the environment as requested, point
 * it at cache_path, initialize the shared region, and report the device-0
 * limit. env_val == NULL means the limit variable is absent for this process
 * (the SSH/PAM / env-stripped case). Returns the observed limit, or UINT64_MAX
 * on failure.
 */
static uint64_t probe_limit(probe_result_t *slot, const char *cache_path,
                            const char *env_val) {
    atomic_store_explicit(&slot->reported, 0, memory_order_relaxed);
    atomic_store_explicit(&slot->done, 0, memory_order_release);

    pid_t child = fork();
    if (child < 0) {
        return UINT64_MAX;
    }
    if (child == 0) {
        if (env_val != NULL) {
            setenv(LIMIT_ENV_KEY, env_val, 1);
        } else {
            unsetenv(LIMIT_ENV_KEY);
        }
        setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, cache_path, 1);
        log_utils_init();
        ensure_initialized();
        atomic_store_explicit(&slot->reported,
                              get_current_device_memory_limit(0),
                              memory_order_relaxed);
        atomic_store_explicit(&slot->done, 1, memory_order_release);
        _exit(0);
    }

    if (wait_child_bounded(child, TEST_TIMEOUT_MS) != 0 ||
        atomic_load_explicit(&slot->done, memory_order_acquire) != 1) {
        return UINT64_MAX;
    }
    return atomic_load_explicit(&slot->reported, memory_order_relaxed);
}

/* Build a unique, non-existent cache path per scenario. */
static int make_cache_path(char *buf, size_t buflen, const char *tag) {
    int n = snprintf(buf, buflen, "/tmp/hami-env-isolation-%d-%s.cache",
                     (int)getpid(), tag);
    if (n < 0 || (size_t)n >= buflen) {
        return -1;
    }
    unlink(buf);
    return 0;
}

struct case_result {
    const char *name;
    uint64_t observed;
    uint64_t expected;
    int fix_flips; /* 1 if an accepted #2125 fix must change expected */
};

static int report(const struct case_result *c) {
    int ok = (c->observed == c->expected);
    printf("  %-52s observed=%-11" PRIu64 " %s%s\n",
           c->name, c->observed, ok ? "OK" : "MISMATCH",
           c->fix_flips ? "   [FIX FLIPS THIS]" : "");
    return ok ? 0 : 1;
}

int main(void) {
    probe_result_t *slot = mmap(NULL, sizeof(*slot), PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (slot == MAP_FAILED) {
        perror("mmap(probe slot)");
        return 1;
    }

    char cache_a[128], cache_c[128], cache_d[128];
    if (make_cache_path(cache_a, sizeof(cache_a), "baseline") != 0 ||
        make_cache_path(cache_c, sizeof(cache_c), "ssh-first") != 0 ||
        make_cache_path(cache_d, sizeof(cache_d), "poison") != 0) {
        fprintf(stderr, "failed to build cache paths\n");
        return 1;
    }

    int failures = 0;

    /* [A] Baseline: configured process seeds a fresh region. Enforcement works. */
    struct case_result a = {
        .name = "[A] configured process, fresh region",
        .observed = probe_limit(slot, cache_a, LIMIT_ENV_VAL),
        .expected = LIMIT_BYTES, .fix_flips = 0};
    failures += report(&a);

    /* [C] SSH/PAM-style clean environment is the first writer of a fresh
     *     region: the limit variable is absent, so the region is seeded with 0
     *     and this process is unlimited. This is the SSH session escaping the
     *     cap on its own. An accepted fix must make this ENFORCED. */
    struct case_result c = {
        .name = "[C] env-stripped process, first writer",
        .observed = probe_limit(slot, cache_c, NULL),
        .expected = 0, .fix_flips = 1};
    failures += report(&c);

    /* [D] First-writer-wins poisoning: an env-stripped process seeds the region
     *     with 0, then the correctly-configured workload joins the SAME region
     *     and also observes 0 -- the library logs "Limit inconsistency detected"
     *     and continues. The pod's own workload loses its limit. An accepted fix
     *     must make the configured process ENFORCED here. */
    uint64_t seed = probe_limit(slot, cache_d, NULL); /* env-less seeds first */
    struct case_result d0 = {
        .name = "[D] env-less seed of shared region",
        .observed = seed, .expected = 0, .fix_flips = 0};
    failures += report(&d0);
    struct case_result d = {
        .name = "[D] configured process joins poisoned region",
        .observed = probe_limit(slot, cache_d, LIMIT_ENV_VAL),
        .expected = 0, .fix_flips = 1};
    failures += report(&d);

    unlink(cache_a);
    unlink(cache_c);
    unlink(cache_d);
    munmap(slot, sizeof(*slot));

    if (failures != 0) {
        fprintf(stderr, "%d env-isolation characterization case(s) mismatched\n",
                failures);
        return 1;
    }
    puts("env-isolation characterization tests passed");
    return 0;
}

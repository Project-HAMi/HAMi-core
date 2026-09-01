/*
 * Reproducer: cross-process cuMemAlloc TOCTOU against the shared memory limit.
 *
 * Vanilla add_chunk() (src/allocator/allocator.c):
 *   1) oom_check(usage + size)   // no inter-process lock / reservation
 *   2) real cuMemAlloc
 *   3) pthread_mutex (process-local) + oom_check again + add_gpu_device_memory_usage
 *
 * When two processes each request less than the limit but the sum exceeds it,
 * both can pass step 1 (and often step 3) before either commits usage, so both
 * allocations succeed and tracked usage overshoots the limit.
 *
 * Build: discovered by test/CMakeLists.txt via ./build.sh
 *
 * Run (GPU + libvgpu required):
 *   mkdir -p /tmp/vgpulock
 *   CACHE=/tmp/hami_oom_race.cache
 *   rm -f "$CACHE"
 *   export LD_PRELOAD=$PWD/build/libvgpu.so
 *   export CUDA_DEVICE_MEMORY_SHARED_CACHE="$CACHE"
 *   export CUDA_DEVICE_MEMORY_LIMIT=1024m
 *   export LIBCUDA_LOG_LEVEL=1
 *   ./build/test/test_concurrent_oom_race
 *
 * Or: ./test/run_concurrent_oom_race.sh
 *   (defaults to --expect-fixed)
 *
 * Suggested sizing: limit=1024m, alloc=600m so each request fits alone
 * (after two process contexts) but 2*alloc exceeds the limit.
 * Exit codes:
 *   0  race reproduced (both concurrent allocs succeeded)  OR  --expect-fixed
 *      mode and limit held for all rounds
 *   1  setup / sequential control failure
 *   2  race not observed within rounds (inconclusive under default mode)
 */

#include <cuda.h>
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

#define CHECK_DRV(call)                                                        \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            const char *_n = NULL;                                             \
            cuGetErrorName(_e, &_n);                                           \
            fprintf(stderr, "FATAL %s:%d: %s -> %d (%s)\n", __FILE__,          \
                    __LINE__, #call, (int)_e, _n ? _n : "?");                  \
            _exit(1);                                                          \
        }                                                                      \
    } while (0)

typedef struct {
    /* barrier: parent sets go; children increment ready / done */
    volatile int ready;
    volatile int go;
    volatile int done;
    volatile int child_ok[2];     /* 1 = cuMemAlloc succeeded */
    volatile int child_res[2];    /* raw CUresult */
    volatile int sequential_ok;   /* control: lone alloc under limit */
    volatile int hold_ok;         /* control: second alloc while first holds */
    volatile int stop;
} shm_state_t;

/* Parse HAMI_RACE_ALLOC / HAMI_RACE_ROUNDS style values.
 * On success writes *out and returns 0. Unset/empty → default.
 * Malformed env (ERANGE, trailing junk, zero) → -1. */
static int parse_size_env(const char *name, size_t default_bytes, size_t *out) {
    const char *v = getenv(name);
    char *end = NULL;
    uint64_t n;
    size_t result;

    if (v == NULL || v[0] == '\0') {
        *out = default_bytes;
        return 0;
    }
    errno = 0;
    n = (uint64_t)strtoull(v, &end, 10);
    if (end == v || errno == ERANGE) {
        return -1;
    }
    if (*end == 'G' || *end == 'g') {
        if (n > (SIZE_MAX >> 30)) {
            return -1;
        }
        result = (size_t)n << 30;
        end++;
    } else if (*end == 'M' || *end == 'm') {
        if (n > (SIZE_MAX >> 20)) {
            return -1;
        }
        result = (size_t)n << 20;
        end++;
    } else if (*end == 'K' || *end == 'k') {
        if (n > (SIZE_MAX >> 10)) {
            return -1;
        }
        result = (size_t)n << 10;
        end++;
    } else {
        result = (size_t)n;
    }
    if (*end != '\0' || result == 0) {
        return -1;
    }
    *out = result;
    return 0;
}

#ifndef WAIT_UNTIL_TIMEOUT_SEC
#define WAIT_UNTIL_TIMEOUT_SEC 60
#endif

/* Returns 0 when *p == want, -1 on timeout (avoids indefinite CI hangs). */
static int wait_until(volatile int *p, int want) {
    struct timespec start, now;

    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        return -1;
    }
    while (*p != want) {
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
            return -1;
        }
        if (now.tv_sec - start.tv_sec >= WAIT_UNTIL_TIMEOUT_SEC) {
            return -1;
        }
        usleep(1000);
    }
    return 0;
}

static void child_init_cuda(CUcontext *ctx) {
    CUdevice device;

    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuDeviceGet(&device, TEST_DEVICE_ID));
#if CUDA_VERSION >= 13000
    CHECK_DRV(cuCtxCreate(ctx, NULL, 0, device));
#else
    CHECK_DRV(cuCtxCreate(ctx, 0, device));
#endif
}

static void child_main(int id, shm_state_t *st, size_t alloc_bytes) {
    CUcontext ctx;
    CUdeviceptr dptr = 0;
    CUresult res;

    child_init_cuda(&ctx);

    /* Signal ready for sequential control / race rounds. */
    __sync_fetch_and_add(&st->ready, 1);

    for (;;) {
        if (wait_until(&st->go, 1) != 0) {
            fprintf(stderr, "child %d: timed out waiting for go=1\n", id);
            _exit(1);
        }
        if (st->stop) {
            break;
        }

        /* Round starts with no outstanding allocation (freed in teardown). */
        dptr = 0;
        __sync_synchronize();
        res = cuMemAlloc(&dptr, alloc_bytes);
        if (res != CUDA_SUCCESS) {
            dptr = 0;
        }
        st->child_res[id] = (int)res;
        st->child_ok[id] = (res == CUDA_SUCCESS) ? 1 : 0;
        __sync_synchronize();

        __sync_fetch_and_add(&st->done, 1);
        if (wait_until(&st->go, 0) != 0) {
            fprintf(stderr, "child %d: timed out waiting for go=0\n", id);
            _exit(1);
        }
        /* Teardown before ready++ so finish_round sees a clean slate. */
        if (dptr != 0) {
            cuMemFree(dptr);
            dptr = 0;
        }
        __sync_fetch_and_add(&st->ready, 1);
    }

    if (dptr != 0) {
        cuMemFree(dptr);
    }
    cuCtxDestroy(ctx);
    _exit(0);
}

static int spawn_children(pid_t kids[2], shm_state_t *st, size_t alloc_bytes) {
    int i;

    for (i = 0; i < 2; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return -1;
        }
        if (pid == 0) {
            child_main(i, st, alloc_bytes);
        }
        kids[i] = pid;
    }
    return 0;
}

static void kill_children(pid_t kids[2]) {
    int i;
    for (i = 0; i < 2; i++) {
        if (kids[i] > 0) {
            kill(kids[i], SIGKILL);
            waitpid(kids[i], NULL, 0);
            kids[i] = -1;
        }
    }
}

static void start_round(shm_state_t *st) {
    st->child_ok[0] = 0;
    st->child_ok[1] = 0;
    st->child_res[0] = -1;
    st->child_res[1] = -1;
    st->done = 0;
    __sync_synchronize();
    st->go = 1;
}

static int finish_round(shm_state_t *st) {
    if (wait_until(&st->done, 2) != 0) {
        return -1;
    }
    /* Zero ready before releasing children. If go=0 comes first, a child may
     * ready++ and then parent clears it — lost increment → hang on ready==2. */
    st->ready = 0;
    __sync_synchronize();
    st->go = 0;
    if (wait_until(&st->ready, 2) != 0) {
        return -1;
    }
    return 0;
}

static const char *cu_res_name(int res) {
    switch ((CUresult)res) {
        case CUDA_SUCCESS:
            return "CUDA_SUCCESS";
        case CUDA_ERROR_OUT_OF_MEMORY:
            return "CUDA_ERROR_OUT_OF_MEMORY";
        case CUDA_ERROR_INVALID_VALUE:
            return "CUDA_ERROR_INVALID_VALUE";
        case CUDA_ERROR_NOT_INITIALIZED:
            return "CUDA_ERROR_NOT_INITIALIZED";
        default:
            return "CUresult(other)";
    }
}

int main(int argc, char **argv) {
    shm_state_t *st;
    pid_t kids[2] = {-1, -1};
    size_t alloc_bytes;
    int rounds;
    int expect_fixed = 0;
    int i;
    int race_hits = 0;
    int both_fail = 0;
    int one_ok = 0;
    const char *limit_env;
    const char *cache_env;
    const char *preload;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--expect-fixed") == 0) {
            expect_fixed = 1;
        }
    }

    preload = getenv("LD_PRELOAD");
    limit_env = getenv("CUDA_DEVICE_MEMORY_LIMIT");
    cache_env = getenv("CUDA_DEVICE_MEMORY_SHARED_CACHE");
    if (parse_size_env("HAMI_RACE_ALLOC", (size_t)600 << 20, &alloc_bytes) != 0) {
        fprintf(stderr,
                "ERROR: invalid HAMI_RACE_ALLOC (use e.g. 600m; no spaces; non-zero)\n");
        return 1;
    }
    {
        size_t rounds_sz;

        if (parse_size_env("HAMI_RACE_ROUNDS", 30, &rounds_sz) != 0 ||
            rounds_sz > (size_t)INT_MAX) {
            fprintf(stderr,
                    "ERROR: invalid HAMI_RACE_ROUNDS (positive integer required)\n");
            return 1;
        }
        rounds = (int)rounds_sz;
    }

    printf("=== HAMi-core concurrent oom_check race reproducer ===\n");
    printf("LD_PRELOAD=%s\n", preload ? preload : "(unset — libvgpu will not hook)");
    printf("CUDA_DEVICE_MEMORY_LIMIT=%s\n", limit_env ? limit_env : "(unset)");
    printf("CUDA_DEVICE_MEMORY_SHARED_CACHE=%s\n",
           cache_env ? cache_env : "(unset → fallback path)");
    printf("alloc_bytes=%zu rounds=%d expect_fixed=%d\n\n", alloc_bytes, rounds,
           expect_fixed);

    if (preload == NULL || preload[0] == '\0') {
        fprintf(stderr, "ERROR: LD_PRELOAD must point at libvgpu.so\n");
        return 1;
    }
    if (limit_env == NULL || limit_env[0] == '\0') {
        fprintf(stderr, "ERROR: set CUDA_DEVICE_MEMORY_LIMIT (e.g. 512m)\n");
        return 1;
    }

    st = mmap(NULL, sizeof(*st), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
              -1, 0);
    if (st == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    memset((void *)st, 0, sizeof(*st));

    /*
     * Sequential control (single process): alloc must succeed under limit.
     * Proves the chosen size individually fits before we race.
     */
    {
        pid_t p = fork();
        if (p < 0) {
            perror("fork");
            return 1;
        }
        if (p == 0) {
            CUcontext ctx;
            CUdeviceptr dptr = 0;
            CUresult res;

            child_init_cuda(&ctx);
            res = cuMemAlloc(&dptr, alloc_bytes);
            st->sequential_ok = (res == CUDA_SUCCESS) ? 1 : 0;
            if (res != CUDA_SUCCESS) {
                fprintf(stderr, "sequential alloc failed: %s\n", cu_res_name(res));
            }
            if (dptr) {
                cuMemFree(dptr);
            }
            cuCtxDestroy(ctx);
            _exit(0);
        }
        waitpid(p, NULL, 0);
    }
    if (!st->sequential_ok) {
        fprintf(stderr,
                "ERROR: single-process alloc of %zu failed under limit=%s.\n"
                "       Lower HAMI_RACE_ALLOC or raise CUDA_DEVICE_MEMORY_LIMIT.\n",
                alloc_bytes, limit_env);
        return 1;
    }
    printf("[control] single-process alloc OK (%zu bytes)\n", alloc_bytes);

    /* Hold control: process A holds alloc; process B must be denied. */
    {
        pid_t a, b;

        st->ready = 0;
        st->go = 0;
        st->done = 0;
        st->hold_ok = 0;

        a = fork();
        if (a < 0) {
            perror("fork");
            return 1;
        }
        if (a == 0) {
            CUcontext ctx;
            CUdeviceptr dptr = 0;

            child_init_cuda(&ctx);
            CHECK_DRV(cuMemAlloc(&dptr, alloc_bytes));
            __sync_fetch_and_add(&st->ready, 1);
            if (wait_until(&st->go, 1) != 0) {
                fprintf(stderr, "hold-control A: timed out waiting for release\n");
                _exit(1);
            }
            cuMemFree(dptr);
            cuCtxDestroy(ctx);
            _exit(0);
        }
        if (wait_until(&st->ready, 1) != 0) {
            fprintf(stderr, "ERROR: timed out waiting for hold-control A ready\n");
            kill(a, SIGKILL);
            waitpid(a, NULL, 0);
            return 1;
        }

        b = fork();
        if (b < 0) {
            perror("fork");
            kill(a, SIGKILL);
            waitpid(a, NULL, 0);
            return 1;
        }
        if (b == 0) {
            CUcontext ctx;
            CUdeviceptr dptr = 0;
            CUresult res;

            child_init_cuda(&ctx);
            res = cuMemAlloc(&dptr, alloc_bytes);
            /* Expect OOM: A already holds ~alloc_bytes against the limit. */
            st->hold_ok = (res == CUDA_ERROR_OUT_OF_MEMORY) ? 1 : 0;
            if (res == CUDA_SUCCESS) {
                fprintf(stderr,
                        "hold-control: B unexpectedly succeeded (%s)\n",
                        cu_res_name(res));
                cuMemFree(dptr);
            } else if (res != CUDA_ERROR_OUT_OF_MEMORY) {
                fprintf(stderr, "hold-control: B got %s (want OOM)\n",
                        cu_res_name(res));
            }
            cuCtxDestroy(ctx);
            _exit(0);
        }
        waitpid(b, NULL, 0);
        st->go = 1;
        waitpid(a, NULL, 0);

        if (!st->hold_ok) {
            fprintf(stderr,
                    "ERROR: when A holds %zu, B should get OOM under limit=%s.\n"
                    "       Pick alloc so 2*alloc > limit (after context overhead).\n",
                    alloc_bytes, limit_env);
            return 1;
        }
        printf("[control] A holds + B denied (OOM) OK — limit enforcement works "
               "when serialized\n\n");
    }

    /* Concurrent race rounds */
    st->ready = 0;
    st->go = 0;
    st->done = 0;
    st->stop = 0;
    if (spawn_children(kids, st, alloc_bytes) != 0) {
        return 1;
    }
    if (wait_until(&st->ready, 2) != 0) {
        fprintf(stderr,
                "ERROR: timed out waiting for race children ready "
                "(ready=%d; child init/CUDA failure?)\n",
                st->ready);
        kill_children(kids);
        return 1;
    }

    for (i = 0; i < rounds; i++) {
        start_round(st);
        if (finish_round(st) != 0) {
            fprintf(stderr,
                    "ERROR: timed out in round %d (done=%d ready=%d)\n", i,
                    st->done, st->ready);
            kill_children(kids);
            return 1;
        }

        if (st->child_ok[0] && st->child_ok[1]) {
            race_hits++;
            printf("[round %d] RACE: both succeeded (res=%s / %s)\n", i,
                   cu_res_name(st->child_res[0]), cu_res_name(st->child_res[1]));
            if (!expect_fixed) {
                /* One hit is enough to prove the bug. */
                break;
            }
        } else if (!st->child_ok[0] && !st->child_ok[1]) {
            both_fail++;
            printf("[round %d] both failed (%s / %s)\n", i,
                   cu_res_name(st->child_res[0]), cu_res_name(st->child_res[1]));
        } else {
            one_ok++;
            printf("[round %d] correctly serialized: ok=%d/%d (%s / %s)\n", i,
                   st->child_ok[0], st->child_ok[1],
                   cu_res_name(st->child_res[0]), cu_res_name(st->child_res[1]));
        }
    }

    st->stop = 1;
    start_round(st);
    /* Children exit on stop; do not wait on done==2 forever if one already died */
    usleep(200000);
    kill_children(kids);

    printf("\n--- summary ---\n");
    printf("race_hits(both success)=%d  one_ok=%d  both_fail=%d\n", race_hits,
           one_ok, both_fail);

    if (expect_fixed) {
        if (race_hits == 0) {
            printf("PASS (--expect-fixed): no dual success across %d rounds\n",
                   rounds);
            return 0;
        }
        printf("FAIL (--expect-fixed): observed %d dual-success race(s)\n",
               race_hits);
        return 1;
    }

    if (race_hits > 0) {
        printf("BUG REPRODUCED: concurrent allocs both passed oom_check while "
               "sum exceeds limit\n");
        return 0;
    }
    printf("INCONCLUSIVE: no dual success in %d rounds.\n"
           "  Retry with more HAMI_RACE_ROUNDS (e.g. 100).\n",
           rounds);
    return 2;
}

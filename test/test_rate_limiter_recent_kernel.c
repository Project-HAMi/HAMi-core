/*
 * GPU-free regression test for rate_limiter() and recent_kernel (issue #350).
 * rate_limiter() must publish recent_kernel=2 even when the utilization
 * switch is off, and must wait while recent_kernel is negative.  The watcher
 * source is included to reach its static limit state; the driver and shared
 * region are stubbed below.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "multiprocess/multiprocess_utilization_watcher.c"  // NOLINT(build/include)

static _Atomic int fake_recent_kernel;

int get_recent_kernel(void) { return fake_recent_kernel; }

int set_recent_kernel(int value) {
    fake_recent_kernel = value;
    return 0;
}

int cas_recent_kernel(int expected, int desired) {
    return atomic_compare_exchange_strong(&fake_recent_kernel, &expected,
                                          desired);
}

CUresult cuCtxGetDevice(CUdevice *device) {
    *device = 0;
    return CUDA_SUCCESS;
}

static int failures;

static void check(int ok, const char *what) {
    printf("%-4s %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

static void *release(void *arg) {
    (void)arg;
    usleep(200 * 1000);
    fake_recent_kernel = 0;
    return NULL;
}

static void run(int sm_limit, int util_switch, int recent) {
    cached_sm_limit[0] = sm_limit;
    cached_util_switch = util_switch;
    fake_recent_kernel = recent;
    g_cur_cuda_cores[0] = 1000;
    rate_limiter(10, 1);
}

int main(void) {
    log_utils_init();

    run(50, 0, 0);
    check(fake_recent_kernel == 2, "switch off: recent_kernel set to 2");
    check(g_cur_cuda_cores[0] == 1000, "switch off: tokens untouched");

    run(50, 0, 1);
    check(fake_recent_kernel == 2, "switch off: decayed value refreshed");

    run(50, 1, 0);
    check(fake_recent_kernel == 2 && g_cur_cuda_cores[0] == 990,
          "switch on: recent_kernel set and tokens consumed");

    run(100, 0, 0);
    check(fake_recent_kernel == 0, "no limit: recent_kernel untouched");
    run(0, 0, 0);
    check(fake_recent_kernel == 0, "limit 0: recent_kernel untouched");

    pthread_t tid;
    struct timespec t0, t1;
    cached_sm_limit[0] = 50;
    cached_util_switch = 0;
    fake_recent_kernel = -1;
    pthread_create(&tid, NULL, release, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    rate_limiter(10, 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    pthread_join(tid, NULL);
    check(t1.tv_sec > t0.tv_sec || t1.tv_nsec - t0.tv_nsec > 150 * 1000000L,
          "negative recent_kernel blocks the caller");
    check(fake_recent_kernel == 2, "recent_kernel set after release");

    printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

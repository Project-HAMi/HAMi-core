/*
 * GPU-free regression test for the SM limiter step (issue #348).
 * The refill step per tick has to be the same fraction of the token bucket on
 * every device, and must stay what it was on an 80 SM device.  The watcher
 * source is included to reach its static state; the driver and shared region
 * are stubbed below.
 */
#include <stdint.h>
#include <stdio.h>

#include "multiprocess/multiprocess_utilization_watcher.c"  // NOLINT(build/include)

int get_recent_kernel(void) { return 0; }
int set_recent_kernel(int value) { (void)value; return 0; }

#define FACTOR_UNDER_TEST 32
#define MAX_THREADS_PER_SM 1536

static int failures;

static void check(int ok, const char *what) {
    printf("%-4s %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

static void setup(int sm_num) {
    g_sm_num[0] = sm_num;
    g_max_thread_per_sm[0] = MAX_THREADS_PER_SM;
    g_total_cuda_cores[0] = (int64_t)MAX_THREADS_PER_SM * sm_num * FACTOR_UNDER_TEST;
}

/* Step per tick, as parts per million of the bucket. */
static int64_t step_ppm(int sm_num, int up_limit, int current) {
    setup(sm_num);
    return delta(up_limit, current, 0, 0) * 1000000 / g_total_cuda_cores[0];
}

static int close_enough(int64_t a, int64_t b) {
    return a - b <= 1 && b - a <= 1;
}

int main(void) {
    /* {limit, current}: the second case takes the accelerated branch. */
    const int cases[][2] = {{50, 30}, {20, 0}};
    int j;

    for (j = 0; j < 2; j++) {
        int64_t small = step_ppm(20, cases[j][0], cases[j][1]);
        int64_t mid = step_ppm(80, cases[j][0], cases[j][1]);
        int64_t large = step_ppm(188, cases[j][0], cases[j][1]);
        char what[80];

        snprintf(what, sizeof(what), "limit %d, current %d: same fraction on 20/80/188 SMs",
                 cases[j][0], cases[j][1]);
        check(close_enough(small, mid) && close_enough(mid, large), what);
    }

    /* Without acceleration the 80 SM step is unchanged from the old formula. */
    setup(80);
    check(delta(50, 30, 0, 0) == 80LL * 80 * MAX_THREADS_PER_SM * 20 / 2560,
          "80 SM step matches the former formula");

    /* Above the limit the same step is taken away. */
    setup(188);
    check(delta(50, 100, g_total_cuda_cores[0], 0) == g_total_cuda_cores[0] - delta(50, 0, 0, 0),
          "above the limit the share shrinks by the same step");

    printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

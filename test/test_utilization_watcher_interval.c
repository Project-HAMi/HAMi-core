/*
 * GPU-free test for the utilization watcher poll interval.
 * The watcher needs 120 ms ticks only when a device has a core limit below
 * 100; otherwise it polls once a second.  The watcher source is included to
 * reach its static helper; the shared region is stubbed below.
 */
#include <stdio.h>
#include <time.h>

#include "multiprocess/multiprocess_utilization_watcher.c"  // NOLINT(build/include)

int get_recent_kernel(void) { return 0; }
int set_recent_kernel(int value) { (void)value; return 0; }

static int failures;

static void check(int ok, const char *what) {
    printf("%-4s %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

static int polls_every(const int *limits, unsigned int count, const struct timespec *want) {
    struct timespec got = {.tv_sec = -1, .tv_nsec = -1};

    choose_watcher_interval(limits, count, &got);
    return got.tv_sec == want->tv_sec && got.tv_nsec == want->tv_nsec;
}

int main(void) {
    const int unlimited[] = {100, 100, 100, 100};
    const int unset[] = {0, 0};
    const int one_limited[] = {100, 100, 50, 100};
    const int lowest[] = {1};
    const int highest[] = {99};
    const int out_of_range[] = {-1, 101};

    check(g_wait.tv_sec == 0 && g_wait.tv_nsec == 120 * MILLISEC, "limited poll is 120 ms");
    check(g_idle_wait.tv_sec == 1 && g_idle_wait.tv_nsec == 0, "idle poll is 1 s");

    check(polls_every(NULL, 0, &g_idle_wait), "no devices: 1 s");
    check(polls_every(unlimited, 4, &g_idle_wait), "limit 100 on every device: 1 s");
    check(polls_every(unset, 2, &g_idle_wait), "limit 0: 1 s");
    check(polls_every(one_limited, 4, &g_wait), "one device at 50: 120 ms");
    check(polls_every(lowest, 1, &g_wait), "limit 1: 120 ms");
    check(polls_every(highest, 1, &g_wait), "limit 99: 120 ms");
    check(polls_every(out_of_range, 2, &g_idle_wait), "limits outside 1..99 do not need the limiter: 1 s");
    check(polls_every(one_limited, 2, &g_idle_wait), "devices past count are ignored: 1 s");

    printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

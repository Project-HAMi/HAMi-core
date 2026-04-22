#include <assert.h>
#include <stdio.h>
#include "../../src/vulkan/throttle_adapter.h"

/* Stub of HAMi-core's rate_limiter so this test links without the full lib. */
static int g_rl_calls = 0;
void rate_limiter(int grids, int blocks) { (void)grids;(void)blocks; g_rl_calls++; }

int main(void) {
    hami_vulkan_throttle();
    hami_vulkan_throttle();
    assert(g_rl_calls == 2);
    printf("ok: adapter forwards to rate_limiter\n");
    return 0;
}

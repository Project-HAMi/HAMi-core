#include "vulkan/throttle_adapter.h"

/* Defined in libvgpu/src/multiprocess/multiprocess_utilization_watcher.c
 * (linked into the same libvgpu.so at final link time). */
extern void rate_limiter(int grids, int blocks);

void hami_vulkan_throttle(void) {
    /* Consume one token — represents "one queue submission". The
     * rate_limiter interprets (grids*blocks) as the claim size; we use
     * the smallest unit (1,1) so Vulkan submits compete fairly with
     * tiny CUDA kernel launches. */
    rate_limiter(1, 1);
}

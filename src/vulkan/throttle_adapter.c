#include "vulkan/throttle_adapter.h"
#include "include/hami_core_export.h"

void hami_vulkan_throttle(void) {
    /* Consume one token — represents "one queue submission". The
     * underlying rate_limiter interprets (grids*blocks) as the claim
     * size; the wrapper uses (1,1) so Vulkan submits compete fairly
     * with tiny CUDA kernel launches. */
    hami_core_throttle();
}

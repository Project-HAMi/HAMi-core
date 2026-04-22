#ifndef HAMI_VK_THROTTLE_ADAPTER_H
#define HAMI_VK_THROTTLE_ADAPTER_H

/* Consume one "compute unit" token from the HAMi-core SM rate limiter.
 * When the HAMi SM limit is 0 or >= 100 (unlimited), this is a no-op
 * inherited from the underlying rate_limiter. Call once per Vulkan
 * vkQueueSubmit/vkQueueSubmit2 before forwarding to the next layer. */
void hami_vulkan_throttle(void);

#endif

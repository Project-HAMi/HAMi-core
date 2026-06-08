#ifndef SRC_VULKAN_THROTTLE_ADAPTER_H_
#define SRC_VULKAN_THROTTLE_ADAPTER_H_

/* Consume one "compute unit" token from the HAMi-core SM rate limiter.
 * When the HAMi SM limit is 0 or >= 100 (unlimited), this is a no-op
 * inherited from the underlying rate_limiter. Call once per Vulkan
 * vkQueueSubmit/vkQueueSubmit2 before forwarding to the next layer. */
void hami_vulkan_throttle(void);

#endif  // SRC_VULKAN_THROTTLE_ADAPTER_H_

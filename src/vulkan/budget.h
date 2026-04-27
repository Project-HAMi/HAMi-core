#ifndef SRC_VULKAN_BUDGET_H_
#define SRC_VULKAN_BUDGET_H_
#include <stddef.h>

/* Reserve `size` bytes on device `dev` for a Vulkan allocation.
 * Returns 1 when the allocation fits the pod budget and the usage
 * counter has been incremented; 0 when the request would exceed the
 * budget (caller must return VK_ERROR_OUT_OF_DEVICE_MEMORY). If the
 * budget is unlimited (HAMi-core limit sentinel == 0), always grants. */
int  hami_budget_reserve(int dev, size_t size);

/* Inverse of a successful reserve — decrements the usage counter. */
void hami_budget_release(int dev, size_t size);

/* Current per-device budget in bytes. Returns 0 when unlimited. */
size_t hami_budget_of(int dev);

#endif  // SRC_VULKAN_BUDGET_H_

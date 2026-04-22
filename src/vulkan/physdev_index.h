#ifndef HAMI_VK_PHYSDEV_INDEX_H
#define HAMI_VK_PHYSDEV_INDEX_H

#include <vulkan/vulkan.h>

/* Resolve the HAMi-core device index for a VkPhysicalDevice by comparing its
 * Vulkan deviceUUID (from VK_KHR_get_physical_device_properties2) against
 * each NVML device UUID. Returns a cached mapping on subsequent calls.
 * Returns -1 if the device could not be resolved (e.g., software rasterizer
 * or NVML unavailable); callers should treat this as "no budget enforcement
 * for this device". */
int hami_vk_physdev_index(VkPhysicalDevice p);

#endif

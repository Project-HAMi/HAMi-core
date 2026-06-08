#ifndef SRC_VULKAN_DISPATCH_H_
#define SRC_VULKAN_DISPATCH_H_

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

typedef struct hami_instance_dispatch {
    VkInstance handle;
    PFN_vkGetInstanceProcAddr next_gipa;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties2 GetPhysicalDeviceMemoryProperties2;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
    PFN_vkEnumerateDeviceLayerProperties EnumerateDeviceLayerProperties;
    struct hami_instance_dispatch *next;
} hami_instance_dispatch_t;

typedef struct hami_device_dispatch {
    VkDevice handle;
    VkPhysicalDevice physical;
    PFN_vkGetDeviceProcAddr next_gdpa;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkQueueSubmit QueueSubmit;
#if defined(VK_VERSION_1_3)
    PFN_vkQueueSubmit2 QueueSubmit2;
#endif
    struct hami_device_dispatch *next;
} hami_device_dispatch_t;

hami_instance_dispatch_t *hami_instance_lookup(VkInstance inst);
hami_instance_dispatch_t *hami_instance_register(VkInstance inst, PFN_vkGetInstanceProcAddr gipa);
void hami_instance_unregister(VkInstance inst);

/* Helper for layer Enumerate hooks: returns the first registered instance
 * dispatch (suitable for forwarding NULL-pLayerName Enumerate* queries to
 * the next layer/ICD). NULL when no instance is registered yet. */
hami_instance_dispatch_t *hami_instance_first(void);

hami_device_dispatch_t *hami_device_lookup(VkDevice dev);
hami_device_dispatch_t *hami_device_register(VkDevice dev, VkPhysicalDevice phys, PFN_vkGetDeviceProcAddr gdpa);
void hami_device_unregister(VkDevice dev);

#endif  // SRC_VULKAN_DISPATCH_H_

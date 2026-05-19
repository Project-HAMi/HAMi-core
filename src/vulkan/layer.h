#ifndef SRC_VULKAN_LAYER_H_
#define SRC_VULKAN_LAYER_H_

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

/* Vulkan-Headers 1.3.280+ dropped VK_LAYER_EXPORT. Default visibility on
 * ELF/Mach-O is sufficient; Windows would need __declspec(dllexport). */
#ifndef VK_LAYER_EXPORT
#  if defined(_WIN32)
#    define VK_LAYER_EXPORT __declspec(dllexport)
#  else
#    define VK_LAYER_EXPORT __attribute__((visibility("default")))
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

VK_LAYER_EXPORT VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct);

PFN_vkVoidFunction VKAPI_CALL
hami_vkGetInstanceProcAddr(VkInstance instance, const char *pName);

PFN_vkVoidFunction VKAPI_CALL
hami_vkGetDeviceProcAddr(VkDevice device, const char *pName);

#ifdef __cplusplus
}
#endif

#endif  // SRC_VULKAN_LAYER_H_

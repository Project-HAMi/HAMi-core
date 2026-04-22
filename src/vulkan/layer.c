#include "layer.h"
#include "dispatch.h"
#include <string.h>
#include <stdlib.h>

/* forward declarations for hooks implemented in sibling files */
extern void hami_vk_hook_instance(hami_instance_dispatch_t *d);
extern void hami_vk_hook_device(hami_device_dispatch_t *d);

static VkLayerInstanceCreateInfo *find_chain_info(const VkInstanceCreateInfo *pCreateInfo,
                                                  VkLayerFunction func) {
    const VkLayerInstanceCreateInfo *ci = pCreateInfo->pNext;
    while (ci) {
        if (ci->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && ci->function == func) {
            return (VkLayerInstanceCreateInfo *)ci;
        }
        ci = (const VkLayerInstanceCreateInfo *)ci->pNext;
    }
    return NULL;
}

static VkLayerDeviceCreateInfo *find_dev_chain_info(const VkDeviceCreateInfo *pCreateInfo,
                                                    VkLayerFunction func) {
    const VkLayerDeviceCreateInfo *ci = pCreateInfo->pNext;
    while (ci) {
        if (ci->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && ci->function == func) {
            return (VkLayerDeviceCreateInfo *)ci;
        }
        ci = (const VkLayerDeviceCreateInfo *)ci->pNext;
    }
    return NULL;
}

static VKAPI_ATTR VkResult VKAPI_CALL
hami_vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkInstance *pInstance) {
    VkLayerInstanceCreateInfo *chain = find_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain || !chain->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    PFN_vkCreateInstance next_create =
        (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult r = next_create(pCreateInfo, pAllocator, pInstance);
    if (r != VK_SUCCESS) return r;

    hami_instance_dispatch_t *d = hami_instance_register(*pInstance, next_gipa);
    hami_vk_hook_instance(d);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
hami_vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    hami_instance_dispatch_t *d = hami_instance_lookup(instance);
    if (d) d->DestroyInstance(instance, pAllocator);
    hami_instance_unregister(instance);
}

static VKAPI_ATTR VkResult VKAPI_CALL
hami_vkCreateDevice(VkPhysicalDevice physicalDevice,
                    const VkDeviceCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkDevice *pDevice) {
    VkLayerDeviceCreateInfo *chain = find_dev_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain || !chain->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   next_gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    PFN_vkCreateDevice next_create =
        (PFN_vkCreateDevice)next_gipa(VK_NULL_HANDLE, "vkCreateDevice");
    VkResult r = next_create(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (r != VK_SUCCESS) return r;

    hami_device_dispatch_t *d = hami_device_register(*pDevice, physicalDevice, next_gdpa);
    hami_vk_hook_device(d);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
hami_vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    hami_device_dispatch_t *d = hami_device_lookup(device);
    if (d) d->DestroyDevice(device, pAllocator);
    hami_device_unregister(device);
}

/* GIPA / GDPA: return our wrappers for hooked names, next-layer for the rest. */

/* Hooked functions implemented in other TUs; declarations here. */
VKAPI_ATTR void VKAPI_CALL hami_vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
VKAPI_ATTR void VKAPI_CALL hami_vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2*);
VKAPI_ATTR VkResult VKAPI_CALL hami_vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*);
VKAPI_ATTR void     VKAPI_CALL hami_vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL hami_vkQueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
VKAPI_ATTR VkResult VKAPI_CALL hami_vkQueueSubmit2(VkQueue, uint32_t, const VkSubmitInfo2*, VkFence);

#define HAMI_HOOK(name) do { if (strcmp(pName, "vk" #name) == 0) return (PFN_vkVoidFunction)hami_vk##name; } while (0)

PFN_vkVoidFunction VKAPI_CALL
hami_vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    HAMI_HOOK(CreateInstance);
    HAMI_HOOK(DestroyInstance);
    HAMI_HOOK(CreateDevice);
    HAMI_HOOK(GetInstanceProcAddr);
    HAMI_HOOK(GetPhysicalDeviceMemoryProperties);
    HAMI_HOOK(GetPhysicalDeviceMemoryProperties2);

    hami_instance_dispatch_t *d = hami_instance_lookup(instance);
    if (!d) return NULL;
    return d->next_gipa(instance, pName);
}

PFN_vkVoidFunction VKAPI_CALL
hami_vkGetDeviceProcAddr(VkDevice device, const char *pName) {
    HAMI_HOOK(DestroyDevice);
    HAMI_HOOK(GetDeviceProcAddr);
    HAMI_HOOK(AllocateMemory);
    HAMI_HOOK(FreeMemory);
    HAMI_HOOK(QueueSubmit);
    HAMI_HOOK(QueueSubmit2);

    hami_device_dispatch_t *d = hami_device_lookup(device);
    if (!d) return NULL;
    return d->next_gdpa(device, pName);
}

VK_LAYER_EXPORT VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct) {
    if (pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (pVersionStruct->loaderLayerInterfaceVersion > 2)
        pVersionStruct->loaderLayerInterfaceVersion = 2;

    pVersionStruct->pfnGetInstanceProcAddr = hami_vkGetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr   = hami_vkGetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}

/* Placeholders — real bodies live in hooks_memory.c / hooks_submit.c.
   Stubs here keep layer.c linkable while the remaining hook TUs
   land over Task 1.3 (allocate/free) and Task 1.5 (submit). */
#ifndef HAMI_VK_HOOKS_PRESENT
void hami_vk_hook_device(hami_device_dispatch_t *d)     { (void)d; }
VKAPI_ATTR VkResult VKAPI_CALL hami_vkAllocateMemory(VkDevice d, const VkMemoryAllocateInfo *i, const VkAllocationCallbacks *a, VkDeviceMemory *m) { (void)d;(void)i;(void)a;(void)m; return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
VKAPI_ATTR void     VKAPI_CALL hami_vkFreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) { (void)d;(void)m;(void)a; }
VKAPI_ATTR VkResult VKAPI_CALL hami_vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo *s, VkFence f) { (void)q;(void)n;(void)s;(void)f; return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL hami_vkQueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *s, VkFence f) { (void)q;(void)n;(void)s;(void)f; return VK_SUCCESS; }
#endif

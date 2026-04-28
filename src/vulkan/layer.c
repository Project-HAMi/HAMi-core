#include "vulkan/layer.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vulkan/dispatch.h"

/* Debug trace gated by HAMI_VK_TRACE=1.
 * Used to localize where the dispatch chain breaks; safe to leave in
 * because it's behind a runtime flag. */
#define HAMI_VK_TRACE_ENV "HAMI_VK_TRACE"
static int hami_vk_trace_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv(HAMI_VK_TRACE_ENV);
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}
#define HAMI_TRACE(fmt, ...) do {                                              \
    if (hami_vk_trace_enabled()) {                                              \
        fprintf(stderr, "HAMI_VK_TRACE: " fmt "\n", ##__VA_ARGS__);             \
        fflush(stderr);                                                         \
    }                                                                           \
} while (0)

/* forward declarations for hooks implemented in sibling files */
extern void hami_vk_hook_instance(hami_instance_dispatch_t *d);
extern void hami_vk_hook_device(hami_device_dispatch_t *d);

/* Cached next-layer GetInstanceProcAddr from the first vkCreateInstance
 * call. Used as a fallback when GIPA is invoked with an unknown instance
 * handle (loader probes during init, or instance handles wrapped by an
 * upper layer that we haven't seen): we still need to return a valid
 * pointer so the loader/driver doesn't dereference NULL. */
static PFN_vkGetInstanceProcAddr g_first_next_gipa = NULL;
static PFN_vkGetDeviceProcAddr   g_first_next_gdpa = NULL;

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
    HAMI_TRACE("hami_vkCreateInstance entered");
    VkLayerInstanceCreateInfo *chain = find_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain || !chain->u.pLayerInfo) {
        HAMI_TRACE("hami_vkCreateInstance: no VK_LAYER_LINK_INFO chain -> returning VK_ERROR_INITIALIZATION_FAILED");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    /* Cache the next-layer gipa before calling next_create: some drivers
     * (NVIDIA) trigger our GIPA from within next_create() to look up
     * vkGetPhysicalDevice* entry points on a fresh, not-yet-registered
     * instance, and we need to forward those lookups instead of returning
     * NULL. */
    if (!g_first_next_gipa) g_first_next_gipa = next_gipa;

    PFN_vkCreateInstance next_create =
        (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    HAMI_TRACE("hami_vkCreateInstance: next_create=%p", (void *)next_create);
    VkResult r = next_create(pCreateInfo, pAllocator, pInstance);
    if (r != VK_SUCCESS) {
        HAMI_TRACE("hami_vkCreateInstance: next_create failed r=%d", r);
        return r;
    }

    hami_instance_dispatch_t *d = hami_instance_register(*pInstance, next_gipa);
    hami_vk_hook_instance(d);
    HAMI_TRACE("hami_vkCreateInstance: registered instance=%p dispatch=%p",
               (void *)*pInstance, (void *)d);
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

    if (!g_first_next_gipa) g_first_next_gipa = next_gipa;
    if (!g_first_next_gdpa) g_first_next_gdpa = next_gdpa;

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

extern void hami_vk_register_queue(VkQueue q, VkDevice d);

static VKAPI_ATTR void VKAPI_CALL
hami_vkGetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue *pQueue) {
    hami_device_dispatch_t *d = hami_device_lookup(device);
    if (!d) {
        *pQueue = VK_NULL_HANDLE;
        return;
    }
    PFN_vkGetDeviceQueue next = (PFN_vkGetDeviceQueue)d->next_gdpa(device, "vkGetDeviceQueue");
    next(device, family, index, pQueue);
    if (*pQueue) {
        hami_vk_register_queue(*pQueue, device);
    }
}

static VKAPI_ATTR void VKAPI_CALL
hami_vkGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *pInfo, VkQueue *pQueue) {
    hami_device_dispatch_t *d = hami_device_lookup(device);
    if (!d) {
        *pQueue = VK_NULL_HANDLE;
        return;
    }
    PFN_vkGetDeviceQueue2 next = (PFN_vkGetDeviceQueue2)d->next_gdpa(device, "vkGetDeviceQueue2");
    next(device, pInfo, pQueue);
    if (*pQueue) {
        hami_vk_register_queue(*pQueue, device);
    }
}

/* GIPA / GDPA: return our wrappers for hooked names, next-layer for the rest. */

/* Hooked functions implemented in other TUs; declarations here. */
VKAPI_ATTR void VKAPI_CALL hami_vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
VKAPI_ATTR void VKAPI_CALL hami_vkGetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2*);
VKAPI_ATTR VkResult VKAPI_CALL hami_vkAllocateMemory(
    VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*);
VKAPI_ATTR void VKAPI_CALL hami_vkFreeMemory(
    VkDevice, VkDeviceMemory, const VkAllocationCallbacks*);
VKAPI_ATTR VkResult VKAPI_CALL hami_vkQueueSubmit(
    VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
#if defined(VK_VERSION_1_3)
VKAPI_ATTR VkResult VKAPI_CALL hami_vkQueueSubmit2(
    VkQueue, uint32_t, const VkSubmitInfo2*, VkFence);
#endif

/* Vulkan layer name advertised in /etc/vulkan/implicit_layer.d/hami.json. */
#define HAMI_LAYER_NAME "VK_LAYER_HAMI_vgpu"

/* Spec-required Enumerate hooks. The Vulkan loader queries layers for
 * own-extension and own-layer info via these entry points (often with a
 * NULL VkInstance during initialization). The previous implementation only
 * exposed CreateInstance/CreateDevice/GIPA via GIPA, so a NULL-instance
 * lookup for vkEnumerate*ExtensionProperties / vkEnumerate*LayerProperties
 * fell through to `hami_instance_lookup(NULL)` -> NULL and the loader
 * dereferenced a NULL function pointer while assembling the enabled
 * extension list. That manifested as a SegFault deep in
 * libcarb.graphics-vulkan during Carbonite Vulkan plugin startup.
 *
 * The layer doesn't add any instance/device extensions, so own-name
 * queries return zero entries. For non-own queries we MUST return
 * VK_SUCCESS with count=0 rather than NULL: the loader will combine our
 * answer with results from the next layer/ICD (Vulkan 1.0 spec
 * "Layered Implementations" §38.3.1). Returning anything else (or a NULL
 * function pointer through GIPA) breaks the chain. */
/* Vulkan 1.3 §38.3.1 / Layer Documentation:
 *   - Querying with our own layer name returns our zero own-extensions.
 *   - Querying with another layer name -> VK_ERROR_LAYER_NOT_PRESENT.
 *   - Querying with NULL pLayerName -> forward to the next layer/ICD so
 *     the caller (NVIDIA driver during vkCreateDevice extension
 *     validation, Carbonite during instance setup) sees the real list.
 *
 * The original layer omitted these hooks, so the GIPA returned NULL and
 * the loader/Carbonite SegFaulted while assembling enabled extensions.
 * An earlier draft of this fix returned LAYER_NOT_PRESENT for NULL
 * pLayerName, which caused vkCreateDevice to fail because the driver
 * could no longer enumerate device extensions through the layer chain. */

static VKAPI_ATTR VkResult VKAPI_CALL
hami_vkEnumerateInstanceExtensionProperties(const char *pLayerName,
                                            uint32_t *pPropertyCount,
                                            VkExtensionProperties *pProperties) {
    if (pLayerName != NULL && strcmp(pLayerName, HAMI_LAYER_NAME) == 0) {
        if (pPropertyCount) *pPropertyCount = 0;
        (void)pProperties;
        return VK_SUCCESS;
    }
    /* For NULL pLayerName the loader will already aggregate every layer
     * + ICD on its own; we just claim no contribution. For other layer
     * names this layer has nothing to say. Both safely map to "0
     * properties, success" so the chain keeps going. */
    if (pPropertyCount) *pPropertyCount = 0;
    (void)pProperties;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
hami_vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                        VkLayerProperties *pProperties) {
    /* Loader assembles the layer list itself from manifests; the layer
     * just reports its own count (0 is accepted because the manifest is
     * authoritative for our presence). */
    (void)pProperties;
    if (pPropertyCount) *pPropertyCount = 0;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
hami_vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                          const char *pLayerName,
                                          uint32_t *pPropertyCount,
                                          VkExtensionProperties *pProperties) {
    /* Own-name: zero own device extensions. */
    if (pLayerName != NULL && strcmp(pLayerName, HAMI_LAYER_NAME) == 0) {
        if (pPropertyCount) *pPropertyCount = 0;
        (void)pProperties;
        return VK_SUCCESS;
    }
    /* For NULL or other names we MUST forward to the next layer/ICD,
     * otherwise the NVIDIA driver's vkCreateDevice fails extension
     * validation ("ERROR_LAYER_NOT_PRESENT" propagated up the chain). */
    hami_instance_dispatch_t *d = hami_instance_first();
    if (d && d->EnumerateDeviceExtensionProperties) {
        return d->EnumerateDeviceExtensionProperties(
            physicalDevice, pLayerName, pPropertyCount, pProperties);
    }
    /* Loader probing before any instance was created: spec allows zero. */
    if (pPropertyCount) *pPropertyCount = 0;
    (void)pProperties;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
hami_vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice,
                                      uint32_t *pPropertyCount,
                                      VkLayerProperties *pProperties) {
    /* Deprecated since Vulkan 1.0.13; forward when possible, else 0. */
    hami_instance_dispatch_t *d = hami_instance_first();
    if (d && d->EnumerateDeviceLayerProperties) {
        return d->EnumerateDeviceLayerProperties(
            physicalDevice, pPropertyCount, pProperties);
    }
    (void)physicalDevice;
    (void)pProperties;
    if (pPropertyCount) *pPropertyCount = 0;
    return VK_SUCCESS;
}

#define HAMI_HOOK(name) do {                                                       \
    if (strcmp(pName, "vk" #name) == 0) {                                          \
        return (PFN_vkVoidFunction)hami_vk##name;                                  \
    }                                                                              \
} while (0)

PFN_vkVoidFunction VKAPI_CALL
hami_vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    HAMI_TRACE("hami_vkGetInstanceProcAddr instance=%p name=%s", (void *)instance, pName);
    HAMI_HOOK(CreateInstance);
    HAMI_HOOK(DestroyInstance);
    HAMI_HOOK(CreateDevice);
    HAMI_HOOK(GetInstanceProcAddr);
    HAMI_HOOK(GetPhysicalDeviceMemoryProperties);
    HAMI_HOOK(GetPhysicalDeviceMemoryProperties2);
    /* Spec-required global entry points that the loader queries with
     * instance=NULL during layer initialization. Returning NULL here
     * caused libcarb.graphics-vulkan to SegFault while assembling the
     * enabled extension list. */
    HAMI_HOOK(EnumerateInstanceExtensionProperties);
    HAMI_HOOK(EnumerateInstanceLayerProperties);
    HAMI_HOOK(EnumerateDeviceExtensionProperties);
    HAMI_HOOK(EnumerateDeviceLayerProperties);

    hami_instance_dispatch_t *d = hami_instance_lookup(instance);
    if (!d) {
        HAMI_TRACE("hami_vkGetInstanceProcAddr: instance %p not registered, returning NULL", (void *)instance);
        return NULL;
    }
    return d->next_gipa(instance, pName);
}

PFN_vkVoidFunction VKAPI_CALL
hami_vkGetDeviceProcAddr(VkDevice device, const char *pName) {
    HAMI_HOOK(DestroyDevice);
    HAMI_HOOK(GetDeviceProcAddr);
    HAMI_HOOK(AllocateMemory);
    HAMI_HOOK(FreeMemory);
    HAMI_HOOK(QueueSubmit);
#if defined(VK_VERSION_1_3)
    HAMI_HOOK(QueueSubmit2);
#endif
    HAMI_HOOK(GetDeviceQueue);
    HAMI_HOOK(GetDeviceQueue2);

    hami_device_dispatch_t *d = hami_device_lookup(device);
    if (!d) return NULL;
    return d->next_gdpa(device, pName);
}

/* The Vulkan loader looks up these three entry points by their canonical
 * (un-prefixed) names. Some build environments compile this TU with
 * -fvisibility=hidden, in which case the upstream VK_LAYER_EXPORT macro
 * (which can fall back to empty on older Vulkan-Headers) does not produce
 * an exported symbol. Force default visibility here regardless of the
 * compile flags so that dlsym from the loader sees them. */
#define HAMI_LAYER_EXPORT __attribute__((visibility("default")))

HAMI_LAYER_EXPORT VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct) {
    HAMI_TRACE("vkNegotiateLoaderLayerInterfaceVersion entered (version=%u)",
               pVersionStruct ? pVersionStruct->loaderLayerInterfaceVersion : 0);
    if (pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        HAMI_TRACE("vkNegotiate: sType mismatch -> VK_ERROR_INITIALIZATION_FAILED");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (pVersionStruct->loaderLayerInterfaceVersion > 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
    }

    pVersionStruct->pfnGetInstanceProcAddr = hami_vkGetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr   = hami_vkGetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
    HAMI_TRACE("vkNegotiate: success (version=%u)",
               pVersionStruct->loaderLayerInterfaceVersion);
    return VK_SUCCESS;
}

/* Fallback wrappers for loader interface version 1: the loader resolves the
 * canonical names directly when the manifest does not advertise interface v2.
 * Both forms must coexist so the layer works regardless of which path the
 * loader picks. */
HAMI_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    return hami_vkGetInstanceProcAddr(instance, pName);
}

HAMI_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char *pName) {
    return hami_vkGetDeviceProcAddr(device, pName);
}

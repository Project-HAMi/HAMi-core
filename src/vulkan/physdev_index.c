#include "vulkan/physdev_index.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vulkan/dispatch.h"

/* Minimal NVML shim — only the symbols we need, resolved from the NVML
 * library already linked into libvgpu.so. Keeping this file independent of
 * the full NVML header avoids coupling with HAMi-core's NVML-override shim. */
typedef void *nvmlDevice_t;
typedef int   nvmlReturn_t;
#define NVML_SUCCESS 0
extern nvmlReturn_t nvmlInit_v2(void);
extern nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count);
extern nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int idx, nvmlDevice_t *dev);
extern nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t dev, char *uuid, unsigned int length);

#define HAMI_VK_UUID_CACHE_SIZE 32

typedef struct {
    VkPhysicalDevice handle;
    int index;
} uuid_cache_entry_t;

static uuid_cache_entry_t g_cache[HAMI_VK_UUID_CACHE_SIZE];
static int g_cache_fill = 0;
static pthread_mutex_t g_cache_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_nvml_initialized = 0;

static int parse_nvml_uuid(const char *s, uint8_t out[16]) {
    /* NVML typically formats as "GPU-590c05ea-a735-d6ce-75cb-c6b06baecf21"
     * (32 hex chars + 4 hyphens after "GPU-" prefix). Skip the prefix and
     * any hyphens, read 16 bytes as hex pairs. */
    if (strncmp(s, "GPU-", 4) == 0) s += 4;
    for (int i = 0; i < 16; i++) {
        while (*s == '-') s++;
        if (!s[0] || !s[1]) return -1;
        unsigned int v;
        if (sscanf(s, "%2x", &v) != 1) return -1;
        out[i] = (uint8_t)v;
        s += 2;
    }
    return 0;
}

static int nvml_index_for_uuid(const uint8_t vk_uuid[16]) {
    if (!g_nvml_initialized) {
        if (nvmlInit_v2() != NVML_SUCCESS) return -1;
        g_nvml_initialized = 1;
    }
    unsigned int count = 0;
    if (nvmlDeviceGetCount_v2(&count) != NVML_SUCCESS) return -1;
    for (unsigned int i = 0; i < count; i++) {
        nvmlDevice_t dev = NULL;
        if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS) continue;
        char uuid_str[96] = {0};
        if (nvmlDeviceGetUUID(dev, uuid_str, sizeof(uuid_str)) != NVML_SUCCESS) continue;
        uint8_t uuid_bin[16];
        if (parse_nvml_uuid(uuid_str, uuid_bin) != 0) continue;
        if (memcmp(uuid_bin, vk_uuid, 16) == 0) return (int)i;
    }
    return -1;
}

/* Defined in dispatch.c (non-static, exported to sibling TUs). */
extern hami_instance_dispatch_t *g_inst_head;

static int resolve_via_vulkan_props(VkPhysicalDevice p, uint8_t out_uuid[16]) {
    /* Walk registered instance dispatches and use whichever next-layer
     * GetPhysicalDeviceProperties2 is available to read the deviceUUID. */
    for (hami_instance_dispatch_t *it = g_inst_head; it; it = it->next) {
        if (!it->next_gipa) continue;
        PFN_vkGetPhysicalDeviceProperties2 get2 =
            (PFN_vkGetPhysicalDeviceProperties2)
                it->next_gipa(it->handle, "vkGetPhysicalDeviceProperties2");
        if (!get2) {
            get2 = (PFN_vkGetPhysicalDeviceProperties2)
                it->next_gipa(it->handle, "vkGetPhysicalDeviceProperties2KHR");
        }
        if (!get2) continue;
        VkPhysicalDeviceIDProperties id = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
            .pNext = NULL,
        };
        VkPhysicalDeviceProperties2 props = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &id,
        };
        get2(p, &props);
        memcpy(out_uuid, id.deviceUUID, 16);
        return 0;
    }
    return -1;
}

int hami_vk_physdev_index(VkPhysicalDevice p) {
    pthread_mutex_lock(&g_cache_lock);
    for (int i = 0; i < g_cache_fill; i++) {
        if (g_cache[i].handle == p) {
            int idx = g_cache[i].index;
            pthread_mutex_unlock(&g_cache_lock);
            return idx;
        }
    }
    pthread_mutex_unlock(&g_cache_lock);

    uint8_t vk_uuid[16];
    int idx = -1;
    if (resolve_via_vulkan_props(p, vk_uuid) == 0) {
        idx = nvml_index_for_uuid(vk_uuid);
    }

    pthread_mutex_lock(&g_cache_lock);
    if (g_cache_fill < HAMI_VK_UUID_CACHE_SIZE) {
        g_cache[g_cache_fill].handle = p;
        g_cache[g_cache_fill].index = idx;
        g_cache_fill++;
    }
    pthread_mutex_unlock(&g_cache_lock);
    return idx;
}

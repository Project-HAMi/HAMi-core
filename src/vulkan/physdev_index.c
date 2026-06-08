#include "vulkan/physdev_index.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vulkan/dispatch.h"

static int hami_vk_trace_enabled_local(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("HAMI_VK_TRACE");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}
#define HAMI_TRACE(fmt, ...) do {                                              \
    if (hami_vk_trace_enabled_local()) {                                        \
        fprintf(stderr, "HAMI_VK_TRACE: " fmt "\n", ##__VA_ARGS__);             \
        fflush(stderr);                                                         \
    }                                                                           \
} while (0)

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

static void hami_log_uuid(const char *tag, const uint8_t u[16]) {
    if (!hami_vk_trace_enabled_local()) return;
    fprintf(stderr, "HAMI_VK_TRACE: %s = "
                    "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
            tag, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
            u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    fflush(stderr);
}

static int is_zero_uuid(const uint8_t u[16]) {
    for (int i = 0; i < 16; i++) {
        if (u[i] != 0) return 0;
    }
    return 1;
}

static int nvml_index_for_uuid(const uint8_t vk_uuid[16]) {
    if (!g_nvml_initialized) {
        nvmlReturn_t r = nvmlInit_v2();
        HAMI_TRACE("nvmlInit_v2 -> %d", (int)r);
        if (r != NVML_SUCCESS) return -1;
        g_nvml_initialized = 1;
    }
    unsigned int count = 0;
    nvmlReturn_t rc = nvmlDeviceGetCount_v2(&count);
    HAMI_TRACE("nvmlDeviceGetCount_v2 -> rc=%d count=%u", (int)rc, count);
    if (rc != NVML_SUCCESS) return -1;
    hami_log_uuid("vk_uuid (target)", vk_uuid);

    /* Fallback: if Vulkan returned a zero deviceUUID (observed on certain
     * NVIDIA driver+container configurations where the VK_KHR_external_memory
     * ID extension does not populate the UUID into pNext'd struct) AND only
     * one NVML device is visible to this container — which is the standard
     * HAMi operating model where the device-plugin assigns one GPU per
     * container — we can safely map to NVML index 0. Multi-GPU containers
     * fall through to strict UUID matching to avoid mis-binding. */
    if (is_zero_uuid(vk_uuid) && count == 1) {
        HAMI_TRACE("nvml_index_for_uuid: vk_uuid all-zero + NVML count==1 "
                   "-> single-GPU fallback idx=0");
        return 0;
    }

    for (unsigned int i = 0; i < count; i++) {
        nvmlDevice_t dev = NULL;
        nvmlReturn_t rh = nvmlDeviceGetHandleByIndex_v2(i, &dev);
        if (rh != NVML_SUCCESS) {
            HAMI_TRACE("nvmlDeviceGetHandleByIndex_v2[%u] -> rc=%d (skip)", i, (int)rh);
            continue;
        }
        char uuid_str[96] = {0};
        nvmlReturn_t ru = nvmlDeviceGetUUID(dev, uuid_str, sizeof(uuid_str));
        if (ru != NVML_SUCCESS) {
            HAMI_TRACE("nvmlDeviceGetUUID[%u] -> rc=%d (skip)", i, (int)ru);
            continue;
        }
        HAMI_TRACE("nvml[%u] uuid_str='%s'", i, uuid_str);
        uint8_t uuid_bin[16];
        if (parse_nvml_uuid(uuid_str, uuid_bin) != 0) {
            HAMI_TRACE("nvml[%u] parse_nvml_uuid FAILED (skip)", i);
            continue;
        }
        hami_log_uuid("nvml uuid_bin", uuid_bin);
        if (memcmp(uuid_bin, vk_uuid, 16) == 0) {
            HAMI_TRACE("nvml[%u] UUID MATCH -> return idx=%d", i, (int)i);
            return (int)i;
        }
    }
    HAMI_TRACE("nvml_index_for_uuid: NO MATCH -> -1");
    return -1;
}

/* Defined in dispatch.c (non-static, exported to sibling TUs). */
extern hami_instance_dispatch_t *g_inst_head;

static int resolve_via_vulkan_props(VkPhysicalDevice p, uint8_t out_uuid[16]) {
    /* Walk registered instance dispatches and use whichever next-layer
     * GetPhysicalDeviceProperties2 is available to read the deviceUUID. */
    int n = 0;
    for (hami_instance_dispatch_t *it = g_inst_head; it; it = it->next) {
        n++;
        if (!it->next_gipa) {
            HAMI_TRACE("resolve_via_vulkan_props inst[%d] NO next_gipa (skip)", n);
            continue;
        }
        PFN_vkGetPhysicalDeviceProperties2 get2 =
            (PFN_vkGetPhysicalDeviceProperties2)
                it->next_gipa(it->handle, "vkGetPhysicalDeviceProperties2");
        if (!get2) {
            get2 = (PFN_vkGetPhysicalDeviceProperties2)
                it->next_gipa(it->handle, "vkGetPhysicalDeviceProperties2KHR");
        }
        if (!get2) {
            HAMI_TRACE("resolve_via_vulkan_props inst[%d] NO get2 (skip)", n);
            continue;
        }
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
        HAMI_TRACE("resolve_via_vulkan_props inst[%d] OK", n);
        return 0;
    }
    HAMI_TRACE("resolve_via_vulkan_props walked %d insts, NO match", n);
    return -1;
}

int hami_vk_physdev_index(VkPhysicalDevice p) {
    pthread_mutex_lock(&g_cache_lock);
    for (int i = 0; i < g_cache_fill; i++) {
        if (g_cache[i].handle == p) {
            int idx = g_cache[i].index;
            pthread_mutex_unlock(&g_cache_lock);
            HAMI_TRACE("physdev_index CACHE HIT physDev=%p -> idx=%d", (void *)p, idx);
            return idx;
        }
    }
    pthread_mutex_unlock(&g_cache_lock);

    uint8_t vk_uuid[16];
    int idx = -1;
    int rv = resolve_via_vulkan_props(p, vk_uuid);
    HAMI_TRACE("physdev_index resolve_via_vulkan_props physDev=%p -> rc=%d", (void *)p, rv);
    if (rv == 0) {
        idx = nvml_index_for_uuid(vk_uuid);
    }
    HAMI_TRACE("physdev_index physDev=%p -> idx=%d", (void *)p, idx);

    pthread_mutex_lock(&g_cache_lock);
    if (g_cache_fill < HAMI_VK_UUID_CACHE_SIZE) {
        g_cache[g_cache_fill].handle = p;
        g_cache[g_cache_fill].index = idx;
        g_cache_fill++;
    }
    pthread_mutex_unlock(&g_cache_lock);
    return idx;
}

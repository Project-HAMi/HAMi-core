#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vulkan/dispatch.h"
#include "vulkan/budget.h"
#include "vulkan/physdev_index.h"

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

static void clamp_heaps(VkPhysicalDevice p, uint32_t *count, VkMemoryHeap *heaps) {
    int dev = hami_vk_physdev_index(p);
    if (dev < 0) return;          /* unresolved (e.g. software rasterizer) */
    size_t budget = hami_budget_of(dev);
    HAMI_TRACE("clamp_heaps dev=%d budget=%zu count=%u", dev, budget, (unsigned)*count);
    if (budget == 0) return;      /* unlimited — preserve reported heap size */
    for (uint32_t i = 0; i < *count; ++i) {
        if ((heaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        if (heaps[i].size > budget) {
            HAMI_TRACE("clamp_heaps[%u] %llu -> %zu", (unsigned)i,
                       (unsigned long long)heaps[i].size, budget);
            heaps[i].size = budget;
        }
    }
}

VKAPI_ATTR void VKAPI_CALL
hami_vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice p,
                                         VkPhysicalDeviceMemoryProperties *out) {
    HAMI_TRACE("hami_vkGetPhysicalDeviceMemoryProperties physDev=%p", (void *)p);
    extern hami_instance_dispatch_t *g_inst_head;
    int n = 0;
    for (hami_instance_dispatch_t *it = g_inst_head; it; it = it->next) {
        n++;
        if (it->GetPhysicalDeviceMemoryProperties) {
            it->GetPhysicalDeviceMemoryProperties(p, out);
            clamp_heaps(p, &out->memoryHeapCount, out->memoryHeaps);
            HAMI_TRACE("hami_vkGetPhysicalDeviceMemoryProperties: clamped via dispatch %d", n);
            return;
        }
    }
    HAMI_TRACE("hami_vkGetPhysicalDeviceMemoryProperties: g_inst_head walked %d entries, no match -> out unmodified", n);
}

VKAPI_ATTR void VKAPI_CALL
hami_vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice p,
                                          VkPhysicalDeviceMemoryProperties2 *out) {
    extern hami_instance_dispatch_t *g_inst_head;
    for (hami_instance_dispatch_t *it = g_inst_head; it; it = it->next) {
        if (it->GetPhysicalDeviceMemoryProperties2) {
            it->GetPhysicalDeviceMemoryProperties2(p, out);
            clamp_heaps(p, &out->memoryProperties.memoryHeapCount,
                        out->memoryProperties.memoryHeaps);
            return;
        }
    }
}

void hami_vk_hook_instance(hami_instance_dispatch_t *d) { (void)d; }

#include "dispatch.h"
#include "budget.h"
#include <stdint.h>
#include <string.h>

/* Provisional device-index heuristic. The plan calls out replacing this
 * with an NVML UUID lookup once the adapter in Task 1.6 lands — for
 * now, a pointer-hash gives a stable per-process index. */
static int physdev_index(VkPhysicalDevice p) {
    return (int)(((uintptr_t)p >> 4) & 0xff);
}

static void clamp_heaps(VkPhysicalDevice p, uint32_t *count, VkMemoryHeap *heaps) {
    size_t budget = hami_budget_of(physdev_index(p));
    if (budget == 0) return;  /* unlimited — preserve reported heap size */
    for (uint32_t i = 0; i < *count; ++i) {
        if ((heaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        if (heaps[i].size > budget) heaps[i].size = budget;
    }
}

VKAPI_ATTR void VKAPI_CALL
hami_vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice p,
                                         VkPhysicalDeviceMemoryProperties *out) {
    extern hami_instance_dispatch_t *g_inst_head;
    for (hami_instance_dispatch_t *it = g_inst_head; it; it = it->next) {
        if (it->GetPhysicalDeviceMemoryProperties) {
            it->GetPhysicalDeviceMemoryProperties(p, out);
            clamp_heaps(p, &out->memoryHeapCount, out->memoryHeaps);
            return;
        }
    }
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

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

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
    HAMI_TRACE("clamp_heaps ENTER physDev=%p count=%u", (void *)p, (unsigned)*count);
    int dev = hami_vk_physdev_index(p);
    HAMI_TRACE("clamp_heaps physdev_index -> dev=%d", dev);
    if (dev < 0) {
        HAMI_TRACE("clamp_heaps EARLY RETURN (dev<0, unresolved physical device)");
        return;
    }
    size_t budget = hami_budget_of(dev);
    HAMI_TRACE("clamp_heaps dev=%d budget=%zu count=%u", dev, budget, (unsigned)*count);
    if (budget == 0) {
        HAMI_TRACE("clamp_heaps EARLY RETURN (budget=0, unlimited)");
        return;
    }
    for (uint32_t i = 0; i < *count; ++i) {
        if ((heaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        if (heaps[i].size > budget) {
            HAMI_TRACE("clamp_heaps[%u] %" PRIu64 " -> %zu", (unsigned)i,
                       (uint64_t)heaps[i].size, budget);
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
    HAMI_TRACE("hami_vkGetPhysicalDeviceMemoryProperties: g_inst_head walked %d entries,"
               " no match -> out unmodified", n);
}

/* Make Carbonite/Kit's "X used / Y available" overlay reflect the pod's
 * partition limit. Earlier attempts to clamp heapBudget directly (to the
 * partition limit, or to limit-usage) caused omni.physx.tensors to dead-
 * lock during plugin initialization, even though heapBudget < heap.size
 * was preserved. PhysX/Carbonite appears to consume heapBudget through
 * paths that go beyond a simple "available = heapBudget - heapUsage"
 * subtraction.
 *
 * Workaround: leave heapBudget untouched (matches what the ICD reports
 * for the host GPU's free memory), and instead inflate heapUsage by the
 * delta (icd_budget - partition_limit). The visible "available" computed
 * by overlay = heapBudget - heapUsage then matches the partition limit,
 * while heapBudget itself stays at the value PhysX expects. */
static void clamp_budget_pnext(VkPhysicalDevice p,
                               const VkMemoryHeap *heaps,
                               uint32_t heap_count,
                               void *pnext_chain) {
    int dev = hami_vk_physdev_index(p);
    if (dev < 0) {
        HAMI_TRACE("clamp_budget_pnext EARLY RETURN (dev<0)");
        return;
    }
    size_t budget = hami_budget_of(dev);
    if (budget == 0) {
        HAMI_TRACE("clamp_budget_pnext EARLY RETURN (budget=0, unlimited)");
        return;
    }
    VkBaseOutStructure *cur = (VkBaseOutStructure *)pnext_chain;
    while (cur) {
        if (cur->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT *bud =
                (VkPhysicalDeviceMemoryBudgetPropertiesEXT *)cur;
            VkDeviceSize budget_vk = (VkDeviceSize)budget;
            for (uint32_t i = 0; i < heap_count && i < VK_MAX_MEMORY_HEAPS; ++i) {
                if ((heaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
                VkDeviceSize icd_budget = bud->heapBudget[i];
                VkDeviceSize icd_usage  = bud->heapUsage[i];
                if (icd_budget <= budget_vk) continue;
                VkDeviceSize delta    = icd_budget - budget_vk;
                VkDeviceSize new_usage = icd_usage + delta;
                if (new_usage > icd_usage) {
                    HAMI_TRACE("clamp_budget_pnext[%u] heapUsage %" PRIu64 " -> %" PRIu64
                               " (limit=%zu icd_budget=%" PRIu64 ")",
                               (unsigned)i,
                               (uint64_t)icd_usage,
                               (uint64_t)new_usage,
                               budget,
                               (uint64_t)icd_budget);
                    bud->heapUsage[i] = new_usage;
                }
            }
            break;
        }
        cur = (VkBaseOutStructure *)cur->pNext;
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
            clamp_budget_pnext(p, out->memoryProperties.memoryHeaps,
                               out->memoryProperties.memoryHeapCount,
                               out->pNext);
            return;
        }
    }
}

void hami_vk_hook_instance(hami_instance_dispatch_t *d) { (void)d; }

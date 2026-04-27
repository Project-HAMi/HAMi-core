#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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

typedef struct mem_entry {
    VkDeviceMemory handle;
    size_t size;
    int dev_idx;
    struct mem_entry *next;
} mem_entry_t;

static mem_entry_t *g_mem_head = NULL;
static pthread_mutex_t g_mem_lock = PTHREAD_MUTEX_INITIALIZER;

static int device_to_index(VkDevice d) {
    hami_device_dispatch_t *dd = hami_device_lookup(d);
    if (!dd) return -1;
    return hami_vk_physdev_index(dd->physical);
}

VKAPI_ATTR VkResult VKAPI_CALL
hami_vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pInfo,
                      const VkAllocationCallbacks *pAlloc, VkDeviceMemory *pMem) {
    HAMI_TRACE("hami_vkAllocateMemory device=%p size=%llu",
               (void *)device, (unsigned long long)pInfo->allocationSize);
    hami_device_dispatch_t *d = hami_device_lookup(device);
    if (!d || !d->AllocateMemory) {
        HAMI_TRACE("hami_vkAllocateMemory: device dispatch missing -> VK_ERROR_INITIALIZATION_FAILED");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    int idx = device_to_index(device);
    if (idx >= 0 && !hami_budget_reserve(idx, pInfo->allocationSize)) {
        HAMI_TRACE("hami_vkAllocateMemory: budget reserve REJECTED idx=%d size=%llu",
                   idx, (unsigned long long)pInfo->allocationSize);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkResult r = d->AllocateMemory(device, pInfo, pAlloc, pMem);
    if (r != VK_SUCCESS) {
        if (idx >= 0) hami_budget_release(idx, pInfo->allocationSize);
        return r;
    }

    mem_entry_t *e = calloc(1, sizeof(*e));
    e->handle = *pMem;
    e->size   = pInfo->allocationSize;
    e->dev_idx = idx;

    pthread_mutex_lock(&g_mem_lock);
    e->next = g_mem_head;
    g_mem_head = e;
    pthread_mutex_unlock(&g_mem_lock);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hami_vkFreeMemory(VkDevice device, VkDeviceMemory mem, const VkAllocationCallbacks *pAlloc) {
    hami_device_dispatch_t *d = hami_device_lookup(device);
    if (d && d->FreeMemory) d->FreeMemory(device, mem, pAlloc);

    pthread_mutex_lock(&g_mem_lock);
    mem_entry_t **pp = &g_mem_head;
    while (*pp && (*pp)->handle != mem) pp = &(*pp)->next;
    if (*pp) {
        mem_entry_t *victim = *pp;
        *pp = victim->next;
        pthread_mutex_unlock(&g_mem_lock);
        if (victim->dev_idx >= 0) hami_budget_release(victim->dev_idx, victim->size);
        free(victim);
        return;
    }
    pthread_mutex_unlock(&g_mem_lock);
}

void hami_vk_hook_device(hami_device_dispatch_t *d) { (void)d; }

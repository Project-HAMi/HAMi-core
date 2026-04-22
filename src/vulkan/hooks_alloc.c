#include "dispatch.h"
#include "budget.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct mem_entry {
    VkDeviceMemory handle;
    size_t size;
    int dev_idx;
    struct mem_entry *next;
} mem_entry_t;

static mem_entry_t *g_mem_head = NULL;
static pthread_mutex_t g_mem_lock = PTHREAD_MUTEX_INITIALIZER;

static int device_to_index(VkDevice d) {
    return (int)(((uintptr_t)d >> 4) & 0xff);
}

VKAPI_ATTR VkResult VKAPI_CALL
hami_vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pInfo,
                      const VkAllocationCallbacks *pAlloc, VkDeviceMemory *pMem) {
    hami_device_dispatch_t *d = hami_device_lookup(device);
    if (!d || !d->AllocateMemory) return VK_ERROR_INITIALIZATION_FAILED;

    int idx = device_to_index(device);
    if (!hami_budget_reserve(idx, pInfo->allocationSize))
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    VkResult r = d->AllocateMemory(device, pInfo, pAlloc, pMem);
    if (r != VK_SUCCESS) {
        hami_budget_release(idx, pInfo->allocationSize);
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
        hami_budget_release(victim->dev_idx, victim->size);
        free(victim);
        return;
    }
    pthread_mutex_unlock(&g_mem_lock);
}

void hami_vk_hook_device(hami_device_dispatch_t *d) { (void)d; }

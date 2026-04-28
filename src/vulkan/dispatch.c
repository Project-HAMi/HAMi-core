#include "vulkan/dispatch.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

hami_instance_dispatch_t *g_inst_head = NULL;
hami_device_dispatch_t   *g_dev_head  = NULL;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void *resolve(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, const char *name) {
    if (!gipa) {
        return NULL;  /* unit-test path: caller fills fn pointers manually */
    }
    return (void *)gipa(inst, name);
}

hami_instance_dispatch_t *hami_instance_register(VkInstance inst, PFN_vkGetInstanceProcAddr gipa) {
    hami_instance_dispatch_t *d = calloc(1, sizeof(*d));
    d->handle = inst;
    d->next_gipa = gipa;
    d->DestroyInstance =
        (PFN_vkDestroyInstance)resolve(gipa, inst, "vkDestroyInstance");
    d->EnumeratePhysicalDevices =
        (PFN_vkEnumeratePhysicalDevices)resolve(gipa, inst, "vkEnumeratePhysicalDevices");
    d->GetPhysicalDeviceMemoryProperties =
        (PFN_vkGetPhysicalDeviceMemoryProperties)resolve(gipa, inst, "vkGetPhysicalDeviceMemoryProperties");
    d->GetPhysicalDeviceMemoryProperties2 =
        (PFN_vkGetPhysicalDeviceMemoryProperties2)resolve(gipa, inst, "vkGetPhysicalDeviceMemoryProperties2");

    pthread_mutex_lock(&g_lock);
    d->next = g_inst_head;
    g_inst_head = d;
    pthread_mutex_unlock(&g_lock);
    return d;
}

hami_instance_dispatch_t *hami_instance_lookup(VkInstance inst) {
    pthread_mutex_lock(&g_lock);
    hami_instance_dispatch_t *p = g_inst_head;
    while (p && p->handle != inst) p = p->next;
    pthread_mutex_unlock(&g_lock);
    return p;
}

void hami_instance_unregister(VkInstance inst) {
    pthread_mutex_lock(&g_lock);
    hami_instance_dispatch_t **pp = &g_inst_head;
    while (*pp && (*pp)->handle != inst) pp = &(*pp)->next;
    if (*pp) {
        hami_instance_dispatch_t *victim = *pp;
        *pp = victim->next;
        free(victim);
    }
    pthread_mutex_unlock(&g_lock);
}

static void *resolve_dev(PFN_vkGetDeviceProcAddr gdpa, VkDevice dev, const char *name) {
    if (!gdpa) {
        return NULL;  /* unit-test path: caller fills fn pointers manually */
    }
    return (void *)gdpa(dev, name);
}

hami_device_dispatch_t *hami_device_register(VkDevice dev, VkPhysicalDevice phys, PFN_vkGetDeviceProcAddr gdpa) {
    hami_device_dispatch_t *d = calloc(1, sizeof(*d));
    d->handle   = dev;
    d->physical = phys;
    d->next_gdpa = gdpa;
    d->DestroyDevice = (PFN_vkDestroyDevice)resolve_dev(gdpa, dev, "vkDestroyDevice");
    d->AllocateMemory = (PFN_vkAllocateMemory)resolve_dev(gdpa, dev, "vkAllocateMemory");
    d->FreeMemory = (PFN_vkFreeMemory)resolve_dev(gdpa, dev, "vkFreeMemory");
    d->QueueSubmit = (PFN_vkQueueSubmit)resolve_dev(gdpa, dev, "vkQueueSubmit");
#if defined(VK_VERSION_1_3)
    d->QueueSubmit2 = (PFN_vkQueueSubmit2)resolve_dev(gdpa, dev, "vkQueueSubmit2");
#endif

    pthread_mutex_lock(&g_lock);
    d->next = g_dev_head;
    g_dev_head = d;
    pthread_mutex_unlock(&g_lock);
    return d;
}

hami_device_dispatch_t *hami_device_lookup(VkDevice dev) {
    pthread_mutex_lock(&g_lock);
    hami_device_dispatch_t *p = g_dev_head;
    while (p && p->handle != dev) p = p->next;
    pthread_mutex_unlock(&g_lock);
    return p;
}

void hami_device_unregister(VkDevice dev) {
    pthread_mutex_lock(&g_lock);
    hami_device_dispatch_t **pp = &g_dev_head;
    while (*pp && (*pp)->handle != dev) pp = &(*pp)->next;
    if (*pp) {
        hami_device_dispatch_t *victim = *pp;
        *pp = victim->next;
        free(victim);
    }
    pthread_mutex_unlock(&g_lock);
}

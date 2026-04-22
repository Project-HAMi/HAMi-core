#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "../../src/vulkan/dispatch.h"

/* Budget adapter stubs (real impl arrives in Task 1.6 / src/vulkan/budget.c). */
static size_t g_used = 0;
static const size_t BUDGET = 1ull << 30; /* 1 GiB */

size_t hami_budget_of(int dev) { (void)dev; return BUDGET; }
int    hami_budget_reserve(int dev, size_t size) {
    (void)dev;
    if (g_used + size > BUDGET) return 0;
    g_used += size;
    return 1;
}
void   hami_budget_release(int dev, size_t size) { (void)dev; g_used -= size; }

/* Throttle stub — hooks_submit.c references it, but this test does not
 * exercise the submit path. */
void hami_vulkan_throttle(void) {}

/* Stub the NVML UUID resolver — always report device 0. */
int hami_vk_physdev_index(VkPhysicalDevice p) { (void)p; return 0; }

static VkResult VKAPI_CALL fake_alloc(VkDevice d, const VkMemoryAllocateInfo *i,
                                      const VkAllocationCallbacks *a, VkDeviceMemory *m) {
    (void)d;(void)a; *m = (VkDeviceMemory)(uintptr_t)(i->allocationSize);
    return VK_SUCCESS;
}
static void VKAPI_CALL fake_free(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) { (void)d;(void)m;(void)a; }

extern VKAPI_ATTR VkResult VKAPI_CALL
hami_vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*);
extern VKAPI_ATTR void VKAPI_CALL
hami_vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*);

int main(void) {
    VkDevice dev = (VkDevice)0x1;
    hami_device_dispatch_t *d = hami_device_register(dev, (VkPhysicalDevice)0x2, NULL);
    d->AllocateMemory = fake_alloc;
    d->FreeMemory     = fake_free;

    VkMemoryAllocateInfo info = { .sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize=(512ull<<20) };
    VkDeviceMemory m1, m2, m3;

    assert(hami_vkAllocateMemory(dev, &info, NULL, &m1) == VK_SUCCESS);
    assert(hami_vkAllocateMemory(dev, &info, NULL, &m2) == VK_SUCCESS);
    assert(hami_vkAllocateMemory(dev, &info, NULL, &m3) == VK_ERROR_OUT_OF_DEVICE_MEMORY);

    hami_vkFreeMemory(dev, m1, NULL);
    assert(hami_vkAllocateMemory(dev, &info, NULL, &m3) == VK_SUCCESS);
    printf("ok: allocate/free budget enforced\n");
    return 0;
}

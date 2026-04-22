#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "../../src/vulkan/dispatch.h"

/* Budget-adapter stubs — real impl in Task 1.6 (src/vulkan/budget.c). */
size_t hami_budget_of(int dev) { (void)dev; return 1ull << 30; /* 1 GiB */ }
int    hami_budget_reserve(int dev, size_t size) { (void)dev; (void)size; return 1; }
void   hami_budget_release(int dev, size_t size) { (void)dev; (void)size; }

/* Throttle stub — hooks_submit.c references it, but this test does not
 * exercise the submit path. */
void hami_vulkan_throttle(void) {}

static void VKAPI_CALL fake_next(VkPhysicalDevice p, VkPhysicalDeviceMemoryProperties *out) {
    (void)p;
    memset(out, 0, sizeof(*out));
    out->memoryHeapCount = 1;
    out->memoryHeaps[0].size = 8ull << 30;
    out->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}

extern VKAPI_ATTR void VKAPI_CALL
hami_vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice p, VkPhysicalDeviceMemoryProperties *out);

int main(void) {
    VkInstance inst = (VkInstance)0x1;
    hami_instance_dispatch_t *d = hami_instance_register(inst, NULL);
    d->GetPhysicalDeviceMemoryProperties = fake_next;

    VkPhysicalDeviceMemoryProperties props;
    hami_vkGetPhysicalDeviceMemoryProperties((VkPhysicalDevice)0x2, &props);
    assert(props.memoryHeapCount == 1);
    assert(props.memoryHeaps[0].size == (1ull << 30));
    printf("ok: heap clamped to 1 GiB\n");
    return 0;
}

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "../../src/vulkan/dispatch.h"

static int g_submit_called = 0;
static VkResult VKAPI_CALL fake_submit(VkQueue q, uint32_t n, const VkSubmitInfo *s, VkFence f) {
    (void)q;(void)n;(void)s;(void)f; g_submit_called++; return VK_SUCCESS;
}

/* Throttle adapter stub — verifies the hook calls the adapter exactly once
 * per submit before forwarding to the next layer. */
static int g_throttle_called = 0;
void hami_vulkan_throttle(void) { g_throttle_called++; }

/* Budget adapter stubs — linked via hooks_alloc.c / hooks_memory.c in this
 * test binary even though we do not exercise allocation here. */
size_t hami_budget_of(int dev) { (void)dev; return 0; }
int    hami_budget_reserve(int dev, size_t size) { (void)dev;(void)size; return 1; }
void   hami_budget_release(int dev, size_t size) { (void)dev;(void)size; }

/* NVML-based physdev resolver stub. */
int hami_vk_physdev_index(VkPhysicalDevice p) { (void)p; return 0; }

extern VKAPI_ATTR VkResult VKAPI_CALL
hami_vkQueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
extern void hami_vk_register_queue(VkQueue q, VkDevice d);

int main(void) {
    VkDevice dev = (VkDevice)0x11;
    VkQueue  q   = (VkQueue)0x22;
    hami_device_dispatch_t *d = hami_device_register(dev, (VkPhysicalDevice)0, NULL);
    d->QueueSubmit = fake_submit;
    hami_vk_register_queue(q, dev);

    VkResult r = hami_vkQueueSubmit(q, 0, NULL, VK_NULL_HANDLE);
    assert(r == VK_SUCCESS);
    assert(g_throttle_called == 1);
    assert(g_submit_called   == 1);
    printf("ok: submit hook throttles then forwards\n");
    return 0;
}

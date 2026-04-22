#include "dispatch.h"
#include "throttle_adapter.h"
#include <pthread.h>
#include <stdlib.h>

/* Queue → Device registry populated by layer.c's vkGetDeviceQueue[2]
 * wrappers (and by unit tests). */
typedef struct q_entry { VkQueue q; VkDevice d; struct q_entry *next; } q_entry_t;
static q_entry_t *g_q_head = NULL;
static pthread_mutex_t g_q_lock = PTHREAD_MUTEX_INITIALIZER;

void hami_vk_register_queue(VkQueue q, VkDevice d) {
    q_entry_t *e = calloc(1, sizeof(*e));
    e->q = q; e->d = d;
    pthread_mutex_lock(&g_q_lock);
    e->next = g_q_head; g_q_head = e;
    pthread_mutex_unlock(&g_q_lock);
}

static VkDevice device_for_queue(VkQueue q) {
    pthread_mutex_lock(&g_q_lock);
    q_entry_t *p = g_q_head;
    while (p && p->q != q) p = p->next;
    VkDevice d = p ? p->d : VK_NULL_HANDLE;
    pthread_mutex_unlock(&g_q_lock);
    return d;
}

VKAPI_ATTR VkResult VKAPI_CALL
hami_vkQueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo *p, VkFence f) {
    VkDevice d = device_for_queue(queue);
    hami_device_dispatch_t *dd = hami_device_lookup(d);
    if (!dd || !dd->QueueSubmit) return VK_ERROR_INITIALIZATION_FAILED;
    hami_vulkan_throttle();
    return dd->QueueSubmit(queue, n, p, f);
}

#if defined(VK_VERSION_1_3)
VKAPI_ATTR VkResult VKAPI_CALL
hami_vkQueueSubmit2(VkQueue queue, uint32_t n, const VkSubmitInfo2 *p, VkFence f) {
    VkDevice d = device_for_queue(queue);
    hami_device_dispatch_t *dd = hami_device_lookup(d);
    if (!dd || !dd->QueueSubmit2) return VK_ERROR_INITIALIZATION_FAILED;
    hami_vulkan_throttle();
    return dd->QueueSubmit2(queue, n, p, f);
}
#endif

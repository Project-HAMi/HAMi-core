/* libvgpu/src/hami_core_export.c */
#include "include/hami_core_export.h"

#include <stdint.h>
#include <stddef.h>

/* Internal HAMi-core symbols. Both libvgpu_vk.so and the wrappers below
 * see the SAME object code linked into libvgpu.so. We make these
 * symbols visible to other .so files only through the wrappers, never
 * directly: that keeps the libvgpu.so→libvgpu_vk.so contract narrow. */
extern int      oom_check(int dev, size_t addon);
extern int      add_gpu_device_memory_usage(int32_t pid, int dev, size_t usage, int type);
extern int      rm_gpu_device_memory_usage(int32_t pid, int dev, size_t usage, int type);
extern uint64_t get_current_device_memory_limit(int dev);
extern void     rate_limiter(int grids, int blocks);

#define HAMI_EXPORT __attribute__((visibility("default")))

HAMI_EXPORT int hami_core_oom_check(int dev, size_t addon) {
    return oom_check(dev, addon);
}

HAMI_EXPORT int hami_core_add_memory_usage(int32_t pid, int dev, size_t usage, int type) {
    return add_gpu_device_memory_usage(pid, dev, usage, type);
}

HAMI_EXPORT int hami_core_rm_memory_usage(int32_t pid, int dev, size_t usage, int type) {
    return rm_gpu_device_memory_usage(pid, dev, usage, type);
}

HAMI_EXPORT uint64_t hami_core_get_memory_limit(int dev) {
    return get_current_device_memory_limit(dev);
}

HAMI_EXPORT void hami_core_throttle(void) {
    rate_limiter(1, 1);
}

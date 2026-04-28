/* libvgpu/src/include/hami_core_export.h */
#ifndef HAMI_CORE_EXPORT_H_
#define HAMI_CORE_EXPORT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* HAMi-core ↔ libvgpu_vk.so contract.
 * These are the only HAMi-core symbols libvgpu_vk.so depends on.
 * libvgpu.so MUST export them with default visibility; libvgpu_vk.so
 * picks them up via DT_NEEDED link at dlopen() time. */

/* Returns 1 if reserving `addon` bytes on device `dev` would exceed the
 * partition limit, else 0. */
int hami_core_oom_check(int dev, size_t addon);

/* Records `usage` bytes of allocation by (pid, dev). type==2 (DEVICE).
 * Returns 0 on success, non-zero on failure. */
int hami_core_add_memory_usage(int32_t pid, int dev, size_t usage, int type);

/* Releases `usage` bytes by (pid, dev). type==2 (DEVICE). 0 = success. */
int hami_core_rm_memory_usage(int32_t pid, int dev, size_t usage, int type);

/* Returns the partition byte-limit for device `dev`, or 0 = unlimited. */
uint64_t hami_core_get_memory_limit(int dev);

/* Consumes one rate-limiter token (claim size = 1*1). */
void hami_core_throttle(void);

#ifdef __cplusplus
}
#endif

#endif  /* HAMI_CORE_EXPORT_H_ */

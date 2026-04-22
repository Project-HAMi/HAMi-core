#include "budget.h"
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>   /* getpid */

/* HAMi-core internal symbols — linked from the same libvgpu.so.
 * See docs/superpowers/plans/notes/hami-core-layout.md for semantics. */
extern int      oom_check(const int dev, size_t addon);                 /* 1 = OOM, 0 = OK */
extern int      add_gpu_device_memory_usage(int32_t pid, int dev,
                                            size_t usage, int type);    /* 0 = success, 1 = failure */
extern int      rm_gpu_device_memory_usage(int32_t pid, int dev,
                                            size_t usage, int type);    /* 0 = success */
extern uint64_t get_current_device_memory_limit(const int dev);          /* 0 = unlimited */

/* HAMi-core CUDA shim init. Populated via the cuInit() → preInit() chain
 * in libvgpu.c; driven there by pthread_once on pre_cuinit_flag. CUDA
 * apps trigger this naturally, but Vulkan-only apps never call cuInit,
 * so oom_check would find the CUDA trampoline table empty. Call cuInit
 * once on first Vulkan allocation to force preInit/postInit to run. */
typedef int CUresult;
extern CUresult cuInit(unsigned int Flags);

static pthread_once_t g_hami_core_init = PTHREAD_ONCE_INIT;
static void hami_core_init_once(void) { (void)cuInit(0); }

/* Matches the type tag used by the existing CUDA allocator path
 * (src/allocator/allocator.c). HAMi-core tracks usage by (pid, dev)
 * regardless of type, so reusing this tag keeps Vulkan and CUDA in the
 * same bucket. */
#define HAMI_MEM_TYPE_DEVICE 2

int hami_budget_reserve(int dev, size_t size) {
    pthread_once(&g_hami_core_init, hami_core_init_once);
    if (get_current_device_memory_limit(dev) == 0) {
        /* Unlimited — skip check, but still bump the counter so metrics
         * remain accurate. add_gpu_device_memory_usage returns 0 on
         * success; treat any failure as OOM (shared region saturated). */
        return add_gpu_device_memory_usage(getpid(), dev, size, HAMI_MEM_TYPE_DEVICE) == 0;
    }
    if (oom_check(dev, size)) return 0;   /* would exceed budget */
    return add_gpu_device_memory_usage(getpid(), dev, size, HAMI_MEM_TYPE_DEVICE) == 0;
}

void hami_budget_release(int dev, size_t size) {
    rm_gpu_device_memory_usage(getpid(), dev, size, HAMI_MEM_TYPE_DEVICE);
}

size_t hami_budget_of(int dev) {
    pthread_once(&g_hami_core_init, hami_core_init_once);
    return (size_t)get_current_device_memory_limit(dev);
}

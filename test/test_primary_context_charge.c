// Copyright 2026 The HAMi Authors.
/* GPU-free regression: a primary context stays charged while it lives.
 * The release hook used to drop the charge on every release, so retain,
 * retain, release left a live context with no charge, and a failed release
 * dropped it too. The fake driver keeps a real reference count. */
#include <stdio.h>

#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "allocator/allocator.h"

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];
size_t context_size = 4096;
int ctx_activate[16];
static long charged;
static int references;
static CUresult release_result = CUDA_SUCCESS;
static int failures;

#define EXPECT(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); \
        failures++; \
    } \
} while (0)

void ensure_hostpid(void) {}
int driver_admission_enabled(void) { return 0; }
int driver_admission_acquire(hostpid_driver_admission_operation_t operation) {
    (void)operation;
    return -1;
}
void driver_admission_release(int *lease_fd) { *lease_fd = -1; }
void driver_admission_cleanup(void *lease_fd) { driver_admission_release(lease_fd); }
void size_primary_context_lazily(int cudadev) { (void)cudadev; }
cuuint64_t graph_context_id(CUcontext ctx) { (void)ctx; return 0; }
void graph_context_prepare_destroy(CUcontext ctx, cuuint64_t id) { (void)ctx; (void)id; }
void graph_context_destroyed(CUcontext ctx, cuuint64_t id) { (void)ctx; (void)id; }
void graph_primary_context(CUdevice dev, CUcontext ctx) { (void)dev; (void)ctx; }
void graph_primary_context_prepare_release(CUdevice dev) { (void)dev; }
void graph_primary_context_released(CUdevice dev, int reset) { (void)dev; (void)reset; }
uint64_t graph_sync_begin(void) { return 0; }
void graph_synchronized(CUcontext ctx, CUstream stream, int whole_context,
                        uint64_t submitted) {
    (void)ctx; (void)stream; (void)whole_context; (void)submitted;
}
int add_gpu_device_memory_usage(int32_t pid, int dev, size_t usage, int type) {
    (void)pid; (void)dev; (void)type;
    charged += (long)usage;
    return 0;
}
int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t usage, int type) {
    (void)pid; (void)dev; (void)type;
    charged -= (long)usage;
    return 0;
}

static CUresult driver_retain(CUcontext *ctx, CUdevice dev) {
    (void)dev;
    *ctx = (CUcontext)1;
    references++;
    return CUDA_SUCCESS;
}
static CUresult driver_release(CUdevice dev) {
    (void)dev;
    if (release_result != CUDA_SUCCESS) return release_result;
    references--;
    return CUDA_SUCCESS;
}
static CUresult driver_get_state(CUdevice dev, unsigned int *flags, int *active) {
    (void)dev;
    *flags = 0;
    *active = references > 0;
    return CUDA_SUCCESS;
}

static void retain(void) {
    CUcontext ctx;
    EXPECT(cuDevicePrimaryCtxRetain(&ctx, 0) == CUDA_SUCCESS);
}

int main(void) {
    log_utils_init();
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRetain)].fn_ptr = driver_retain;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRelease_v2)].fn_ptr = driver_release;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxGetState)].fn_ptr = driver_get_state;

    retain();
    retain();
    EXPECT(charged == 4096);   /* charged once for two references */
    EXPECT(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
    EXPECT(charged == 4096);   /* one reference stays, so the charge stays */
    EXPECT(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
    EXPECT(charged == 0);      /* the last release frees it */

    retain();
    release_result = CUDA_ERROR_INVALID_VALUE;
    EXPECT(cuDevicePrimaryCtxRelease_v2(0) == CUDA_ERROR_INVALID_VALUE);
    EXPECT(charged == 4096);   /* a failed release keeps the charge */
    release_result = CUDA_SUCCESS;
    EXPECT(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
    EXPECT(charged == 0);

    retain();                  /* a context retained again is charged again */
    EXPECT(charged == 4096);
    EXPECT(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
    EXPECT(charged == 0);

    if (failures) return 1;
    puts("primary context charge tests passed");
    return 0;
}

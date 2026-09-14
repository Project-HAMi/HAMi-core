#ifndef TEST_CONTEXT_HOOK_FIXTURE_H_
#define TEST_CONTEXT_HOOK_FIXTURE_H_

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define TEST_CONTEXT_BYTES 104857600UL

/* Driver and shared-region boundaries for the production context hooks. */
size_t context_size = TEST_CONTEXT_BYTES;
int pidfound = 1;
pthread_once_t post_cuinit_flag = PTHREAD_ONCE_INIT;
static unsigned int driver_refs[CUDA_DEVICE_MAX_COUNT];
static size_t context_charge[CUDA_DEVICE_MAX_COUNT];
static int fail_add;
static int fail_remove;
static CUresult driver_error;
static unsigned int out_of_range_driver_calls;

static CUresult fake_retain(CUcontext *ctx, CUdevice dev) {
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        out_of_range_driver_calls++;
        return CUDA_SUCCESS;
    }
    if (driver_error != CUDA_SUCCESS) {
        return driver_error;
    }
    driver_refs[dev]++;
    *ctx = (CUcontext)(uintptr_t)(dev + 1);
    return CUDA_SUCCESS;
}

static CUresult fake_release(CUdevice dev) {
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        out_of_range_driver_calls++;
        return CUDA_SUCCESS;
    }
    if (driver_error != CUDA_SUCCESS) {
        return driver_error;
    }
    if (driver_refs[dev] == 0) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    driver_refs[dev]--;
    return CUDA_SUCCESS;
}

static CUresult fake_reset(CUdevice dev) {
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        out_of_range_driver_calls++;
        return CUDA_SUCCESS;
    }
    if (driver_error != CUDA_SUCCESS) {
        return driver_error;
    }
    /* Reset destroys resources without releasing retained usage. */
    return CUDA_SUCCESS;
}

cuda_entry_t cuda_library_entry[OVERRIDE_cuCtxSynchronize + 1] = {
    [OVERRIDE_cuDevicePrimaryCtxRetain] = {.fn_ptr = (void *)fake_retain},
    [OVERRIDE_cuDevicePrimaryCtxRelease_v2] = {.fn_ptr = (void *)fake_release},
    [OVERRIDE_cuDevicePrimaryCtxReset_v2] = {.fn_ptr = (void *)fake_reset},
};

int add_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid;
    assert(type == 0);
    assert(dev >= 0 && dev < CUDA_DEVICE_MAX_COUNT);
    assert(bytes > 0);
    if (fail_add) {
        return -1;
    }
    context_charge[dev] += bytes;
    return 0;
}

int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid;
    assert(type == 0);
    assert(dev >= 0 && dev < CUDA_DEVICE_MAX_COUNT);
    assert(bytes > 0);
    if (fail_remove) {
        return -1;
    }
    assert(context_charge[dev] >= bytes);
    context_charge[dev] -= bytes;
    return 0;
}

#endif  // TEST_CONTEXT_HOOK_FIXTURE_H_

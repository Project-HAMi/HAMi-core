// Copyright 2026 The HAMi Authors.
/* GPU-free regression coverage for CUDA array memory-limit accounting. */
#include <stdint.h>
#include <stdio.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

CUresult cuArrayCreate_v2(CUarray *arr, const CUDA_ARRAY_DESCRIPTOR *desc);
CUresult cuArray3DCreate_v2(CUarray *arr, const CUDA_ARRAY3D_DESCRIPTOR *desc);
CUresult cuArrayDestroy(CUarray arr);

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];

static CUresult reserve_result = CUDA_SUCCESS;
static CUresult create_result = CUDA_SUCCESS;
static CUresult create_3d_result = CUDA_SUCCESS;
static CUresult destroy_result = CUDA_SUCCESS;
static int reserve_calls;
static int release_calls;
static int create_calls;
static int create_3d_calls;
static int destroy_calls;
static int failures;
static CUdevice reserved_device;
static size_t reserved_bytes;
static CUdevice released_device;
static size_t released_bytes;
static uintptr_t next_array = 0x1000;

#define EXPECT(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        failures++; \
    } \
} while (0)

void ensure_initialized(void) {}
int is_gpu_core_limit_enabled(void) { return 0; }
int wait_status_self(int status) { return status == 1; }
CUresult cuCtxGetDevice(CUdevice *device) {
    *device = 3;
    return CUDA_SUCCESS;
}

int reserve_device_memory(CUdevice device, size_t bytes) {
    reserve_calls++;
    reserved_device = device;
    reserved_bytes = bytes;
    return reserve_result;
}

void release_device_memory(CUdevice device, size_t bytes) {
    release_calls++;
    released_device = device;
    released_bytes = bytes;
}

static CUresult driver_array_create(CUarray *arr, const CUDA_ARRAY_DESCRIPTOR *desc) {
    (void)desc;
    create_calls++;
    if (create_result == CUDA_SUCCESS) {
        *arr = (CUarray)next_array++;
    }
    return create_result;
}

static CUresult driver_array_3d_create(CUarray *arr,
                                       const CUDA_ARRAY3D_DESCRIPTOR *desc) {
    (void)desc;
    create_3d_calls++;
    if (create_3d_result == CUDA_SUCCESS) {
        *arr = (CUarray)next_array++;
    }
    return create_3d_result;
}

static CUresult driver_array_destroy(CUarray arr) {
    (void)arr;
    destroy_calls++;
    return destroy_result;
}

static void reset_calls(void) {
    reserve_calls = 0;
    release_calls = 0;
    create_calls = 0;
    create_3d_calls = 0;
    destroy_calls = 0;
    reserve_result = CUDA_SUCCESS;
    create_result = CUDA_SUCCESS;
    create_3d_result = CUDA_SUCCESS;
    destroy_result = CUDA_SUCCESS;
}

static void expect_accounting(size_t bytes, int reserve_count, int release_count) {
    EXPECT(reserve_calls == reserve_count);
    EXPECT(release_calls == release_count);
    if (reserve_count != 0) {
        EXPECT(reserved_device == 3);
        EXPECT(reserved_bytes == bytes);
    }
    if (release_count != 0) {
        EXPECT(released_device == 3);
        EXPECT(released_bytes == bytes);
    }
}

static void test_array_create_and_destroy(void) {
    CUDA_ARRAY_DESCRIPTOR desc = {4, 2, CU_AD_FORMAT_UNSIGNED_INT16, 2};
    CUarray array;

    reset_calls();
    EXPECT(cuArrayCreate_v2(&array, &desc) == CUDA_SUCCESS);
    EXPECT(create_calls == 1);
    expect_accounting(32, 1, 0);

    EXPECT(cuArrayDestroy(array) == CUDA_SUCCESS);
    EXPECT(destroy_calls == 1);
    expect_accounting(32, 1, 1);

    EXPECT(cuArrayDestroy(array) == CUDA_SUCCESS);
    EXPECT(destroy_calls == 2);
    expect_accounting(32, 1, 1);
}

static void test_3d_array_create_and_destroy(void) {
    CUDA_ARRAY3D_DESCRIPTOR desc = {2, 3, 4, CU_AD_FORMAT_UNSIGNED_INT16, 2, 0};
    CUarray array;

    reset_calls();
    EXPECT(cuArray3DCreate_v2(&array, &desc) == CUDA_SUCCESS);
    EXPECT(create_3d_calls == 1);
    expect_accounting(96, 1, 0);

    EXPECT(cuArrayDestroy(array) == CUDA_SUCCESS);
    EXPECT(destroy_calls == 1);
    expect_accounting(96, 1, 1);
}

static void test_limit_and_driver_failures(void) {
    CUDA_ARRAY_DESCRIPTOR desc = {4, 2, CU_AD_FORMAT_UNSIGNED_INT16, 2};
    CUarray array;

    reset_calls();
    reserve_result = CUDA_ERROR_OUT_OF_MEMORY;
    EXPECT(cuArrayCreate_v2(&array, &desc) == CUDA_ERROR_OUT_OF_MEMORY);
    EXPECT(create_calls == 0);
    expect_accounting(32, 1, 0);

    reset_calls();
    create_result = CUDA_ERROR_OUT_OF_MEMORY;
    EXPECT(cuArrayCreate_v2(&array, &desc) == CUDA_ERROR_OUT_OF_MEMORY);
    EXPECT(create_calls == 1);
    expect_accounting(32, 1, 1);
}

static void test_destroy_failure_keeps_accounting(void) {
    CUDA_ARRAY_DESCRIPTOR desc = {4, 2, CU_AD_FORMAT_UNSIGNED_INT16, 2};
    CUarray array;

    reset_calls();
    EXPECT(cuArrayCreate_v2(&array, &desc) == CUDA_SUCCESS);
    destroy_result = CUDA_ERROR_INVALID_VALUE;
    EXPECT(cuArrayDestroy(array) == CUDA_ERROR_INVALID_VALUE);
    expect_accounting(32, 1, 0);

    destroy_result = CUDA_SUCCESS;
    EXPECT(cuArrayDestroy(array) == CUDA_SUCCESS);
    expect_accounting(32, 1, 1);
}

static void test_invalid_descriptors(void) {
    CUDA_ARRAY_DESCRIPTOR valid = {4, 2, CU_AD_FORMAT_UNSIGNED_INT16, 2};
    CUDA_ARRAY_DESCRIPTOR invalid_channels = {4, 2, CU_AD_FORMAT_UNSIGNED_INT16, 3};
    CUDA_ARRAY_DESCRIPTOR overflow = {SIZE_MAX, 1, CU_AD_FORMAT_UNSIGNED_INT16, 2};
    CUDA_ARRAY3D_DESCRIPTOR invalid_format = {1, 1, 1, (CUarray_format)33, 1, 0};
    CUarray array;

    reset_calls();
    EXPECT(cuArrayCreate_v2(NULL, &valid) == CUDA_ERROR_INVALID_VALUE);
    EXPECT(cuArrayCreate_v2(&array, NULL) == CUDA_ERROR_INVALID_VALUE);
    EXPECT(cuArrayCreate_v2(&array, &invalid_channels) == CUDA_ERROR_INVALID_VALUE);
    EXPECT(cuArrayCreate_v2(&array, &overflow) == CUDA_ERROR_INVALID_VALUE);
    EXPECT(cuArray3DCreate_v2(&array, &invalid_format) == CUDA_ERROR_INVALID_VALUE);
    EXPECT(create_calls == 0);
    EXPECT(create_3d_calls == 0);
    expect_accounting(0, 0, 0);
}

int main(void) {
    log_utils_init();
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuArrayCreate_v2)].fn_ptr = driver_array_create;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuArray3DCreate_v2)].fn_ptr = driver_array_3d_create;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuArrayDestroy)].fn_ptr = driver_array_destroy;

    test_array_create_and_destroy();
    test_3d_array_create_and_destroy();
    test_limit_and_driver_failures();
    test_destroy_failure_keeps_accounting();
    test_invalid_descriptors();

    printf("CUDA array accounting: %d failures\n", failures);
    return failures != 0;
}

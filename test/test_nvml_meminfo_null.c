// Copyright 2026 The HAMi Authors.
/*
 * GPU-free regression test for nvmlDeviceGetMemoryInfo and
 * nvmlDeviceGetMemoryInfo_v2 with a NULL output pointer (issue #443).
 */
#include <stdint.h>
#include <stdio.h>

#include "include/libnvml_hook.h"

#define TEST_DEVICE ((nvmlDevice_t)0x1)
#define DRV_TOTAL (UINT64_C(24) << 30)
#define DRV_FREE (UINT64_C(20) << 30)
#define TEST_LIMIT (UINT64_C(8) << 30)
#define TEST_USAGE (UINT64_C(2) << 30)

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device,
                                        nvmlMemory_v2_t *memory);

unsigned int nvml_to_cuda_map(unsigned int nvmldev) { return nvmldev; }
unsigned int cuda_to_nvml_map(unsigned int cudadev) { return cudadev; }
uint64_t get_current_device_memory_usage(const int dev) {
    (void)dev;
    return TEST_USAGE;
}
uint64_t get_current_device_memory_monitor(const int dev) {
    (void)dev;
    return TEST_USAGE;
}
uint64_t get_current_device_memory_limit(const int dev) {
    (void)dev;
    return TEST_LIMIT;
}

static int drv_v1_calls;
static int drv_v2_calls;

static nvmlReturn_t stub_get_index(nvmlDevice_t device, unsigned int *index) {
    (void)device;
    *index = 0;
    return NVML_SUCCESS;
}

static nvmlReturn_t stub_mem_info_v1(nvmlDevice_t device,
                                     nvmlMemory_t *memory) {
    (void)device;
    drv_v1_calls++;
    memory->total = DRV_TOTAL;
    memory->free = DRV_FREE;
    memory->used = DRV_TOTAL - DRV_FREE;
    return NVML_SUCCESS;
}

static nvmlReturn_t stub_mem_info_v2(nvmlDevice_t device,
                                     nvmlMemory_v2_t *memory) {
    (void)device;
    drv_v2_calls++;
    memory->total = DRV_TOTAL;
    memory->free = DRV_FREE;
    memory->used = DRV_TOTAL - DRV_FREE;
    memory->reserved = 0;
    return NVML_SUCCESS;
}

static int failures;

#define EXPECT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); \
        failures++; \
    } \
} while (0)

int main(void) {
    nvmlMemory_t m1 = {0};
    nvmlMemory_v2_t m2 = {0};

    log_utils_init();
    nvml_library_entry[NVML_OVERRIDE_ENUM(nvmlDeviceGetIndex)].fn_ptr =
        stub_get_index;
    nvml_library_entry[NVML_OVERRIDE_ENUM(nvmlDeviceGetMemoryInfo)].fn_ptr =
        stub_mem_info_v1;
    nvml_library_entry[NVML_OVERRIDE_ENUM(nvmlDeviceGetMemoryInfo_v2)].fn_ptr =
        stub_mem_info_v2;

    EXPECT(nvmlDeviceGetMemoryInfo(TEST_DEVICE, NULL) ==
           NVML_ERROR_INVALID_ARGUMENT);
    EXPECT(drv_v1_calls == 0);

    EXPECT(nvmlDeviceGetMemoryInfo_v2(TEST_DEVICE, NULL) ==
           NVML_ERROR_INVALID_ARGUMENT);
    EXPECT(drv_v2_calls == 0);

    EXPECT(nvmlDeviceGetMemoryInfo(TEST_DEVICE, &m1) == NVML_SUCCESS);
    EXPECT(drv_v1_calls == 1);
    EXPECT(m1.total == TEST_LIMIT);
    EXPECT(m1.used == TEST_USAGE);
    EXPECT(m1.free == TEST_LIMIT - TEST_USAGE);

    EXPECT(nvmlDeviceGetMemoryInfo_v2(TEST_DEVICE, &m2) == NVML_SUCCESS);
    EXPECT(drv_v2_calls == 1);
    EXPECT(m2.total == TEST_LIMIT);
    EXPECT(m2.used == TEST_USAGE);
    EXPECT(m2.free == TEST_LIMIT - TEST_USAGE);

    if (failures != 0) {
        return 1;
    }
    printf("nvml meminfo null tests passed\n");
    return 0;
}

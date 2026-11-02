// Copyright 2026 The HAMi Authors.
/* GPU-free regression: cuDeviceTotalMem must report the smaller of the memory
 * limit and the real device total. It used to return the limit as is, so an
 * unset limit gave 0 and a limit above the device gave more than exists. */
#include <stdio.h>

#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];
static uint64_t limit;
static CUresult driver_result;
static int failures;

#define EXPECT(x) do { \
    if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); failures++; } \
} while (0)

void ensure_initialized(void) {}
uint64_t get_current_device_memory_limit(const int dev) { (void)dev; return limit; }
static CUresult driver_total(size_t *bytes, CUdevice dev) {
    (void)dev;
    if (driver_result == CUDA_SUCCESS && bytes) *bytes = 1000;
    return driver_result;
}

static size_t total(void) {
    size_t bytes = 12345;
    EXPECT(cuDeviceTotalMem_v2(&bytes, 0) == CUDA_SUCCESS);
    return bytes;
}

int main(void) {
    log_utils_init();
    cuda_library_entry[OVERRIDE_cuDeviceTotalMem_v2].fn_ptr = driver_total;
    limit = 0;    EXPECT(total() == 1000);   /* no limit: real total */
    limit = 600;  EXPECT(total() == 600);    /* limit below the device */
    limit = 5000; EXPECT(total() == 1000);   /* limit above the device */
    EXPECT(cuDeviceTotalMem_v2(NULL, 0) == CUDA_SUCCESS);   /* no write through NULL */
    driver_result = CUDA_ERROR_INVALID_DEVICE;              /* driver errors pass through */
    size_t bytes = 12345;
    EXPECT(cuDeviceTotalMem_v2(&bytes, 9) == CUDA_ERROR_INVALID_DEVICE && bytes == 12345);
    printf("cuDeviceTotalMem: %d failures\n", failures);
    return failures != 0;
}

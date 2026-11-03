/*
 * GPU-free regression test for cuDeviceTotalMem_v2.
 *
 * A zero HAMi memory limit means unlimited memory.  The hook must therefore
 * preserve the CUDA driver's physical-memory result in that case.  When a
 * limit is configured, it must cap the driver's result without claiming more
 * memory than the device has.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define TEST_DEV 0
#define DRIVER_TOTAL (UINT64_C(24) << 30)
#define HAMi_LIMIT (UINT64_C(8) << 30)

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];

static CUresult driver_result = CUDA_SUCCESS;
static int driver_calls;
static int failures;

static CUresult stub_device_total_mem(size_t *bytes, CUdevice dev) {
    driver_calls++;
    if (driver_result != CUDA_SUCCESS) {
        return driver_result;
    }
    if (bytes == NULL || dev != TEST_DEV) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *bytes = DRIVER_TOTAL;
    return CUDA_SUCCESS;
}

static void expect_result(const char *what, CUresult got, CUresult want) {
    if (got != want) {
        fprintf(stderr, "%s: got %d, expected %d\n", what, got, want);
        failures++;
    }
}

static void expect_size(const char *what, size_t got, uint64_t want) {
    if (got != want) {
        fprintf(stderr, "%s: got %" PRIu64 ", expected %" PRIu64 "\n",
                what, (uint64_t)got, want);
        failures++;
    }
}

static void check_limit(const char *what, size_t limit, uint64_t expected) {
    size_t bytes = 0;

    if (set_current_device_memory_limit(TEST_DEV, limit) != 0) {
        fprintf(stderr, "%s: could not configure memory limit\n", what);
        failures++;
        return;
    }
    expect_result(what, cuDeviceTotalMem_v2(&bytes, TEST_DEV), CUDA_SUCCESS);
    expect_size(what, bytes, expected);
}

int main(void) {
    char cache_path[] = "/tmp/hami-device-total-mem.XXXXXX";
    int cache_fd = mkstemp(cache_path);
    size_t bytes = 123;

    if (cache_fd < 0) {
        perror("mkstemp(shared-region cache)");
        return 1;
    }
    close(cache_fd);
    unlink(cache_path);
    if (setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, cache_path, 1) != 0 ||
        setenv("LIBCUDA_LOG_LEVEL", "0", 1) != 0) {
        perror("setenv");
        return 1;
    }

    log_utils_init();
    ensure_initialized();
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDeviceTotalMem_v2)].fn_ptr =
        (void *)stub_device_total_mem;

    check_limit("no limit returns physical memory", 0, DRIVER_TOTAL);
    check_limit("configured limit caps physical memory", HAMi_LIMIT, HAMi_LIMIT);
    check_limit("limit above physical memory stays physical", DRIVER_TOTAL * 2,
                DRIVER_TOTAL);

    driver_result = CUDA_ERROR_UNKNOWN;
    expect_result("driver errors are preserved",
                  cuDeviceTotalMem_v2(&bytes, TEST_DEV), CUDA_ERROR_UNKNOWN);
    expect_size("driver error leaves output untouched", bytes, 123);
    driver_result = CUDA_SUCCESS;

    expect_result("null output is handled by the driver",
                  cuDeviceTotalMem_v2(NULL, TEST_DEV), CUDA_ERROR_INVALID_VALUE);

    if (driver_calls != 5) {
        fprintf(stderr, "driver call count: got %d, expected 5\n", driver_calls);
        failures++;
    }

    unlink(cache_path);
    if (failures != 0) {
        fprintf(stderr, "%d device-total-memory assertion(s) failed\n", failures);
        return 1;
    }
    puts("device total memory tests passed");
    return 0;
}

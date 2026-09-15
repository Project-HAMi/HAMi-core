/*
 * GPU-free regression test for device-id bounds checking in the shared-region
 * accessors.
 *
 * On a thread with no current CUDA context, cuCtxGetDevice() returns
 * CUDA_ERROR_INVALID_CONTEXT and leaves its output parameter unwritten, so an
 * undefined device id can reach the per-device shared-region arrays (limit[],
 * sm_limit[], used[], ...), each sized [CUDA_DEVICE_MAX_COUNT].  Indexing those
 * arrays with an out-of-range id is an out-of-bounds read or write and crashes
 * with SIGSEGV.  Every accessor must reject an id outside
 * [0, CUDA_DEVICE_MAX_COUNT) and return a safe value instead of indexing.
 *
 * This test feeds a battery of out-of-range ids (including the exact garbage
 * value observed in the field) into every guarded accessor and asserts a safe
 * return with no crash.  It also confirms the guard leaves the normal in-range
 * path untouched.  It invokes no CUDA/NVML entry point; section garbage
 * collection drops the GPU-facing production code at link time.
 */
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

/*
 * A device id observed in a field crash report when cuCtxGetDevice() left its
 * output unwritten: uninitialized stack garbage reused as an array index.
 */
#define FIELD_GARBAGE_DEV 1716861768

static const int kIllegalDevs[] = {
    -1,
    -1000000,
    CUDA_DEVICE_MAX_COUNT,
    CUDA_DEVICE_MAX_COUNT + 1,
    1000,
    FIELD_GARBAGE_DEV,
};
#define ILLEGAL_DEV_COUNT ((int)(sizeof(kIllegalDevs) / sizeof(kIllegalDevs[0])))

static int failures;

static void expect_eq(const char *what, int dev, int64_t got, int64_t want) {
    if (got != want) {
        fprintf(stderr, "%s(dev=%d): got %" PRId64 ", expected %" PRId64 "\n",
                what, dev, got, want);
        failures++;
    }
}

static void check_illegal_dev(int dev) {
    /* int-returning accessor: -1 error sentinel. */
    expect_eq("get_current_device_sm_limit", dev,
              get_current_device_sm_limit(dev), -1);
    /*
     * Write path: must refuse with -1 and not store into limit[dev].  This is
     * the out-of-bounds *write* that corrupts memory in the field.
     */
    expect_eq("set_current_device_memory_limit", dev,
              set_current_device_memory_limit(dev, 4096), -1);
    /* uint64_t / size_t getters: 0 == "no limit / no usage", safe downstream. */
    expect_eq("get_current_device_memory_limit", dev,
              (int64_t)get_current_device_memory_limit(dev), 0);
    /* Exercises get_gpu_memory_monitor() transitively. */
    expect_eq("get_current_device_memory_monitor", dev,
              (int64_t)get_current_device_memory_monitor(dev), 0);
    /* Exercises get_gpu_memory_usage() transitively. */
    expect_eq("get_current_device_memory_usage", dev,
              (int64_t)get_current_device_memory_usage(dev), 0);
    expect_eq("get_gpu_memory_usage", dev,
              (int64_t)get_gpu_memory_usage(dev), 0);
}

static void check_valid_dev(void) {
    /* The guard must not disturb the normal in-range path. */
    const int dev = 0;
    const uint64_t limit = 123456789;

    if (set_current_device_memory_limit(dev, limit) != 0) {
        fprintf(stderr,
                "set_current_device_memory_limit(dev=0) failed on the valid path\n");
        failures++;
    }
    if (get_current_device_memory_limit(dev) != limit) {
        fprintf(stderr,
                "get_current_device_memory_limit(dev=0) did not read back the stored limit\n");
        failures++;
    }
    /* These must return without crashing for a valid id. */
    (void)get_current_device_sm_limit(dev);
    (void)get_current_device_memory_monitor(dev);
    (void)get_current_device_memory_usage(dev);
    (void)get_gpu_memory_usage(dev);
}

int main(void) {
    char cache_path[] = "/tmp/hami-device-id-guard.XXXXXX";
    int cache_fd;
    int i;

    cache_fd = mkstemp(cache_path);
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

    for (i = 0; i < ILLEGAL_DEV_COUNT; i++) {
        check_illegal_dev(kIllegalDevs[i]);
    }
    check_valid_dev();

    unlink(cache_path);
    if (failures != 0) {
        fprintf(stderr, "%d device-id guard assertion(s) failed\n", failures);
        return 1;
    }
    puts("device-id guard tests passed");
    return 0;
}

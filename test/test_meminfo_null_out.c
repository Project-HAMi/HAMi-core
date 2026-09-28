/*
 * GPU-free regression test for cuMemGetInfo_v2 with a NULL out pointer.
 *
 * libnvoptix calls cuMemGetInfo_v2(NULL, &total) during OptiX init. The driver
 * accepts that; the hook used to dereference both pointers and segfault
 * (issue #333). The NVML path already guards for it.
 *
 * The test installs its own cuda_library_entry table and its own
 * cuCtxGetDevice, so it needs no driver and no GPU.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define TEST_DEV 0
#define DRV_TOTAL (UINT64_C(24) << 30)
#define DRV_FREE (UINT64_C(20) << 30)
#define TEST_LIMIT (UINT64_C(8) << 30)

/* The production hook, which no header declares. */
CUresult cuMemGetInfo_v2(size_t *free, size_t *total);

/* Stands in for the loaded libcuda.so.1 entry table. */
cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];

/* CUDA_VISIBLE_DEVICES is unset under ctest, so the map is the identity. */
unsigned int cuda_to_nvml_map(unsigned int cudadev) {
    return cudadev;
}

CUresult cuCtxGetDevice(CUdevice *device) {
    *device = TEST_DEV;
    return CUDA_SUCCESS;
}

static CUresult stub_mem_get_info(size_t *free, size_t *total) {
    *free = DRV_FREE;
    *total = DRV_TOTAL;
    return CUDA_SUCCESS;
}

static int failures;

static void expect(const char *what, uint64_t got, uint64_t want) {
    if (got != want) {
        fprintf(stderr, "%s: got %" PRIu64 ", expected %" PRIu64 "\n",
                what, got, want);
        failures++;
    }
}

/* Every call below used to dereference both pointers, so reaching the end of
   this function at all is most of the assertion. */
static void check(const char *what, size_t limit, size_t expect_free,
                  size_t expect_total) {
    size_t free_only = 0, total_only = 0, both_free = 0, both_total = 0;
    char label[128];

    if (set_current_device_memory_limit(TEST_DEV, limit) != 0) {
        fprintf(stderr, "set_current_device_memory_limit failed\n");
        failures++;
        return;
    }

    snprintf(label, sizeof(label), "%s: total only", what);
    printf("  %s\n", label);
    if (cuMemGetInfo_v2(NULL, &total_only) != CUDA_SUCCESS) {
        fprintf(stderr, "%s returned an error\n", label);
        failures++;
    }
    expect(label, total_only, expect_total);

    snprintf(label, sizeof(label), "%s: free only", what);
    printf("  %s\n", label);
    if (cuMemGetInfo_v2(&free_only, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "%s returned an error\n", label);
        failures++;
    }
    expect(label, free_only, expect_free);

    snprintf(label, sizeof(label), "%s: neither", what);
    printf("  %s\n", label);
    if (cuMemGetInfo_v2(NULL, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "%s returned an error\n", label);
        failures++;
    }

    snprintf(label, sizeof(label), "%s: both", what);
    printf("  %s\n", label);
    if (cuMemGetInfo_v2(&both_free, &both_total) != CUDA_SUCCESS) {
        fprintf(stderr, "%s returned an error\n", label);
        failures++;
    }
    expect(label, both_free, expect_free);
    expect(label, both_total, expect_total);
}

int main(void) {
    char cache_path[] = "/tmp/hami-meminfo-null-out.XXXXXX";
    int cache_fd = mkstemp(cache_path);

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

    /* Unbuffered so the last line printed names the call that crashed. */
    setbuf(stdout, NULL);
    log_utils_init();
    ensure_initialized();
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuMemGetInfo_v2)].fn_ptr =
        (void *)stub_mem_get_info;

    /* No limit: total is the driver's, free is reduced by our own usage,
       which is zero in this process. */
    check("no limit", 0, DRV_TOTAL, DRV_TOTAL);
    /* With a limit below the card, both are clamped to it. */
    check("limited", TEST_LIMIT, TEST_LIMIT, TEST_LIMIT);

    unlink(cache_path);
    if (failures != 0) {
        fprintf(stderr, "%d meminfo assertion(s) failed\n", failures);
        return 1;
    }
    puts("meminfo null out-pointer tests passed");
    return 0;
}

/*
 * GPU-free regression test for the OOM check's device index.
 *
 * used[] is NVML indexed, because the slow paths let one process write
 * another's slot and every process in a container shares the region, so two
 * tasks with different CUDA_VISIBLE_DEVICES subsets must not collide.
 * limit[] comes from CUDA_DEVICE_MEMORY_LIMIT_<i> and is CUDA indexed.
 * oom_check read used[] with the CUDA index, so nothing it charged was ever
 * visible to it again and the limit went unenforced.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocator/allocator.h"
#include "multiprocess/multiprocess_memory_limit.h"

/* CUDA_VISIBLE_DEVICES=2,3 */
static const unsigned int kNvmlOf[] = {2, 3};
#define VISIBLE_COUNT ((int)(sizeof(kNvmlOf) / sizeof(kNvmlOf[0])))
#define LIMIT (64u << 20)

unsigned int cuda_to_nvml_map(unsigned int cudadev) {
    if (cudadev < (unsigned int)VISIBLE_COUNT)
        return kNvmlOf[cudadev];
    return cudadev;
}

/* oom_check_impl only calls this for dev == -1, which the test never uses. */
CUresult cuCtxGetDevice(CUdevice *device) {
    *device = 0;
    return CUDA_SUCCESS;
}

static int failures;

static void expect(const char *what, int cudadev, uint64_t got, uint64_t want) {
    if (got != want) {
        fprintf(stderr, "%s(dev=%d): got %" PRIu64 ", expected %" PRIu64 "\n",
                what, cudadev, got, want);
        failures++;
    }
}

static void check_dev(int cudadev) {
    const int nvmldev = (int)cuda_to_nvml_map((unsigned int)cudadev);

    if (set_current_device_memory_limit(cudadev, LIMIT) != 0) {
        fprintf(stderr, "set_current_device_memory_limit(dev=%d) failed\n", cudadev);
        failures++;
        return;
    }

    /* Half the limit fits. */
    expect("oom_check under limit", cudadev, oom_check(cudadev, LIMIT / 2), 0);
    if (reserve_device_memory(cudadev, LIMIT / 2) != 0) {
        fprintf(stderr, "reserve_device_memory(dev=%d) failed\n", cudadev);
        failures++;
        return;
    }

    /* The charge lands in the NVML slot. */
    expect("get_gpu_memory_usage", nvmldev, get_gpu_memory_usage(nvmldev), LIMIT / 2);
    /* And the CUDA slot stays empty, so a test that only compared the two
       would pass on an identity map without proving anything. */
    expect("get_gpu_memory_usage", cudadev, get_gpu_memory_usage(cudadev), 0);

    /* The check has to see that charge, otherwise the limit never binds. */
    expect("oom_check over limit", cudadev, oom_check(cudadev, LIMIT), 1);

    release_device_memory(cudadev, LIMIT / 2);
    expect("get_gpu_memory_usage", nvmldev, get_gpu_memory_usage(nvmldev), 0);
    expect("oom_check after release", cudadev, oom_check(cudadev, LIMIT / 2), 0);
}

int main(void) {
    char cache_path[] = "/tmp/hami-memory-index-map.XXXXXX";
    int cache_fd = mkstemp(cache_path);
    int i;

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

    for (i = 0; i < VISIBLE_COUNT; i++) {
        check_dev(i);
    }

    unlink(cache_path);
    if (failures != 0) {
        fprintf(stderr, "%d device index assertion(s) failed\n", failures);
        return 1;
    }
    puts("memory index map tests passed");
    return 0;
}

// Regression test for Project-HAMi/HAMi-core issue #310: remove_chunk_async()
// used cuCtxGetDevice() -- the CALLING THREAD's currently-bound context's
// device -- to decide which device's tracked usage counter to decrement,
// instead of val->entry->dev, the device the chunk was actually allocated on
// and accounted against (see account_async_chunk() in
// src/allocator/allocator.c, which stores it there via
// INIT_ALLOCATED_LIST_ENTRY at allocation time).
//
// cuMemFreeAsync(dptr, hStream) takes an explicit stream, not an implicit
// "current device" -- CUDA does not require the calling thread's current
// context to match the stream's/allocation's device. A legitimate multi-GPU
// caller (a worker pool that round-robins cuCtxSetCurrent, async cleanup
// running with a different device's context current, etc.) can trip this,
// corrupting device accounting for TWO devices in one call:
//   - the real allocation's device is never decremented -> its tracked
//     usage stays permanently inflated even though the memory was freed;
//   - the wrong device is decremented for memory it never held -> its
//     tracked usage underflows (these are unsigned counters), wrapping to a
//     huge bogus value.
//
// Mechanism verified here: links src/multiprocess/multiprocess_memory_limit.c
// directly into this test binary (the same pattern test_postinit_owner_death.c
// already uses -- see test/CMakeLists.txt) so the test can call
// get_gpu_memory_usage() itself to read the same tracked per-device usage
// counters that account_async_chunk()/remove_chunk_async() mutate via
// add_gpu_device_memory_usage()/rm_gpu_device_memory_usage(). That state
// lives in a plain mmap'd file (see MULTIPROCESS_SHARED_REGION_CACHE_ENV),
// so this directly-linked copy and libvgpu.so's own separately-linked copy
// of the same source file (loaded via LD_PRELOAD) observe the SAME
// underlying counters, despite being two separate static instances of the
// same C code, as long as they agree on the cache file path -- which is why
// CUDA_DEVICE_MEMORY_SHARED_CACHE is pinned below to a path unique to this
// process, making the test hermetic against any file another test run left
// at the default /tmp/cudevshr.cache.
//
// Needs >= 2 CUDA devices. If fewer are visible, the test SKIPs (CTest
// SKIP_RETURN_CODE, see test/CMakeLists.txt) with a clear message rather
// than failing, so it doesn't break single-GPU CI/dev runs.
//
// Reproduces the exact interleaving from the bug report:
//   1. With device 0's context current, allocate on device 0's default pool
//      via the real hooked cuMemAllocAsync (tracked in device_allocasync,
//      entry->dev == 0).
//   2. Switch the current context to device 1.
//   3. Free that pointer via the real hooked cuMemFreeAsync, passing
//      device 0's stream -- current context is device 1, the allocation's
//      tracked device is device 0: exactly the mismatch
//      remove_chunk_async() mishandled.
//   4. Assert device 0's tracked usage dropped back to its pre-allocation
//      baseline (pre-fix: it doesn't) and device 1's tracked usage is
//      unchanged (pre-fix: it underflows to a huge value).
//   5. Allocate again, this time on device 1 with device 1 current (the
//      non-buggy pattern), and confirm device 1's usage now reflects a
//      normal, uncorrupted increment from its baseline -- pre-fix, device 1
//      never recovers from the earlier bogus underflow, so this also fails.
//
// To confirm this test actually exercises the fix: temporarily revert the
// remove_chunk_async() change in src/allocator/allocator.c, rebuild, and
// re-run -- it must fail.
//
// Run standalone (from a build directory with the vgpu target already
// built):
//   ctest -R alloc_async_free_wrong_device --output-on-failure

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define ALLOC_BYTES (32ull * 1024 * 1024) /* 32 MiB */

/* Matches the CTest SKIP_RETURN_CODE configured in test/CMakeLists.txt for
 * this test -- fewer than 2 GPUs is an environment limitation, not a test
 * failure. */
#define SKIP_EXIT_CODE 125

/* Abort-on-error for setup calls that must succeed for the test to be valid. */
#define CHECK_DRV(call)                                                        \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            const char *_n = NULL;                                             \
            cuGetErrorName(_e, &_n);                                           \
            fprintf(stderr, "FATAL %s:%d: %s -> %d (%s)\n", __FILE__,          \
                    __LINE__, #call, (int)_e, _n ? _n : "?");                  \
            return 2;                                                          \
        }                                                                      \
    } while (0)

static void describe(const char *label, CUresult res) {
    const char *name = NULL;
    CUresult q = cuGetErrorName(res, &name);
    if (q != CUDA_SUCCESS || name == NULL)
        printf("  %-40s -> %d  <unrecognized error code>\n", label, (int)res);
    else
        printf("  %-40s -> %d  (%s)\n", label, (int)res, name);
}

int main(void) {
    /* Isolate this run's shared accounting state from anything another test
     * run may have left at the default cache path, so the usage counters
     * read below start from a known baseline. */
    char cache_path[256];
    snprintf(cache_path, sizeof(cache_path),
             "/tmp/cudevshr_test_wrongdevice_%d.cache", (int)getpid());
    setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, cache_path, 1);

    CHECK_DRV(cuInit(0));

    int dev_count = 0;
    CHECK_DRV(cuDeviceGetCount(&dev_count));
    if (dev_count < 2) {
        printf("SKIP: test_alloc_async_free_wrong_device needs >= 2 CUDA "
               "devices to reproduce a cross-device free (found %d)\n",
               dev_count);
        return SKIP_EXIT_CODE;
    }

    CUdevice dev0, dev1;
    CHECK_DRV(cuDeviceGet(&dev0, 0));
    CHECK_DRV(cuDeviceGet(&dev1, 1));

    CUcontext ctx0, ctx1;
#if CUDA_VERSION >= 13000
    CHECK_DRV(cuCtxCreate(&ctx0, NULL, 0, dev0));
    CHECK_DRV(cuCtxCreate(&ctx1, NULL, 0, dev1));
#else
    CHECK_DRV(cuCtxCreate(&ctx0, 0, dev0));
    CHECK_DRV(cuCtxCreate(&ctx1, 0, dev1));
#endif

    /* cuCtxCreate() makes the new context current; make device 0's current
     * again before creating its stream. */
    CHECK_DRV(cuCtxSetCurrent(ctx0));
    CUstream stream0;
    CHECK_DRV(cuStreamCreate(&stream0, CU_STREAM_NON_BLOCKING));

    int failures = 0;

    size_t usage0_before = get_gpu_memory_usage((int)dev0);
    size_t usage1_before = get_gpu_memory_usage((int)dev1);

    printf("[1] cuMemAllocAsync on device 0 (device 0 current)\n");
    CUdeviceptr d = 0;
    CHECK_DRV(cuMemAllocAsync(&d, ALLOC_BYTES, stream0));
    CHECK_DRV(cuStreamSynchronize(stream0));

    size_t usage0_after_alloc = get_gpu_memory_usage((int)dev0);
    size_t alloc_delta = usage0_after_alloc - usage0_before;
    printf("  device0 usage before=%zu after_alloc=%zu (delta=%zu)\n",
           usage0_before, usage0_after_alloc, alloc_delta);
    /* This is the very first async allocation in the process, so the
     * default pool's reservation high-water mark cannot yet be below
     * ALLOC_BYTES; account_async_chunk() must have tracked exactly
     * ALLOC_BYTES. If it didn't, the accounting path this test exercises
     * isn't behaving as expected in this environment -- fail loudly rather
     * than silently passing on a vacuous (zero-delta) case. */
    if (alloc_delta != ALLOC_BYTES) {
        fprintf(stderr,
                "FAIL: expected device0 usage to rise by exactly %llu after "
                "the first allocation, got a delta of %zu -- the accounting "
                "path is not behaving as this test assumes\n",
                ALLOC_BYTES, alloc_delta);
        failures++;
    }

    printf("[2] cuCtxSetCurrent(device 1)\n");
    CHECK_DRV(cuCtxSetCurrent(ctx1));

    printf("[3] cuMemFreeAsync(device-0 pointer, device-0 stream) with device 1 current\n");
    CUresult fres = cuMemFreeAsync(d, stream0);
    describe("cuMemFreeAsync", fres);
    if (fres != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: cross-device free was rejected by the driver "
                        "(expected to succeed per the CUDA API contract)\n");
        failures++;
    }

    /* Switch back to device 0's context to synchronize its stream. */
    CHECK_DRV(cuCtxSetCurrent(ctx0));
    CHECK_DRV(cuStreamSynchronize(stream0));

    size_t usage0_after_free = get_gpu_memory_usage((int)dev0);
    size_t usage1_after_free = get_gpu_memory_usage((int)dev1);
    printf("  device0 usage after_free=%zu (expected %zu)\n",
           usage0_after_free, usage0_before);
    printf("  device1 usage after_free=%zu (expected %zu, unchanged)\n",
           usage1_after_free, usage1_before);

    if (usage0_after_free != usage0_before) {
        fprintf(stderr,
                "FAIL: device0 usage did not return to baseline after free "
                "(%zu != %zu) -- the real allocation's device was not "
                "decremented (issue #310)\n",
                usage0_after_free, usage0_before);
        failures++;
    }
    if (usage1_after_free != usage1_before) {
        fprintf(stderr,
                "FAIL: device1 usage changed from baseline after freeing a "
                "device-0 allocation (%zu != %zu) -- the wrong device was "
                "decremented (issue #310)\n",
                usage1_after_free, usage1_before);
        failures++;
    }

    printf("[5] cuMemAllocAsync on device 1 (device 1 current)\n");
    CUstream stream1;
    CHECK_DRV(cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING));
    CUdeviceptr d1 = 0;
    CHECK_DRV(cuMemAllocAsync(&d1, ALLOC_BYTES, stream1));
    CHECK_DRV(cuStreamSynchronize(stream1));

    size_t usage1_after_second_alloc = get_gpu_memory_usage((int)dev1);
    size_t second_alloc_delta = usage1_after_second_alloc - usage1_before;
    printf("  device1 usage before=%zu after_second_alloc=%zu (delta=%zu, expected %llu)\n",
           usage1_before, usage1_after_second_alloc, second_alloc_delta, ALLOC_BYTES);
    if (second_alloc_delta != ALLOC_BYTES) {
        fprintf(stderr,
                "FAIL: device1 usage did not rise by exactly %llu from its "
                "pre-test baseline after a clean allocation -- device1's "
                "counter is still corrupted from the earlier wrong-device "
                "decrement (issue #310)\n",
                ALLOC_BYTES);
        failures++;
    }

    /* Best-effort teardown. */
    cuMemFreeAsync(d1, stream1);
    cuStreamSynchronize(stream1);
    cuStreamDestroy(stream1);
    cuStreamDestroy(stream0);
    cuCtxDestroy(ctx1);
    cuCtxDestroy(ctx0);

    printf("\nResult: %s\n",
           failures ? "FAIL - device accounting corrupted by cross-device free (issue #310)"
                    : "PASS - cross-device free accounted against the correct device");
    return failures ? 1 : 0;
}

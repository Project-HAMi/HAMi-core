// Fix-verification reproducer for Project-HAMi/HAMi-core issue #2864: async
// CUDA allocation leaks on a post-allocation driver-call failure.
//
// Mechanism (src/allocator/allocator.c on main, pre-fix):
//   add_chunk_async() calls cuMemAllocAsync() (succeeds, real GPU memory is
//   now live), then calls cuDeviceGetMemPool() to look up the device's
//   default pool so the allocation can be accounted against its high-water
//   limit. account_async_chunk() (shared by add_chunk_async() and
//   add_chunk_from_pool_async()) then calls cuMemPoolGetAttribute() to read
//   CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH. Pre-fix, if EITHER of those two
//   driver calls returns anything other than CUDA_SUCCESS, the function
//   returns immediately WITHOUT:
//     (a) freeing the GPU memory that cuMemAllocAsync/cuMemAllocFromPoolAsync
//         already handed back (no cuMemFreeAsync on e->entry->address), and
//     (b) freeing e's heap allocations (e, e->entry, e->entry->allocHandle),
//         since e was never linked into device_allocasync for later cleanup.
//   The caller sees a failure CUresult and reasonably assumes no allocation
//   happened, so it never calls cuMemFreeAsync itself -> the device
//   allocation is leaked permanently, on every occurrence.
//
// Unlike an earlier version of this test, this one does not call
// cuDeviceGetMemPool/cuMemPoolGetAttribute itself with bad arguments -- doing
// that only fails the test's own direct driver calls, not the allocator's
// internal ones, so it could not actually catch a regression here. Instead
// it:
//   1. Drives the allocation through the real hooked entry points
//      (cuMemAllocAsync / cuMemAllocFromPoolAsync), which this process picks
//      up from libvgpu.so via LD_PRELOAD -- exactly as a real application
//      would, so the calls go through allocator.c's add_chunk_async() /
//      add_chunk_from_pool_async() for real.
//   2. Forces the *allocator's* internal cuDeviceGetMemPool /
//      cuMemPoolGetAttribute calls (not the test's) to fail on the very next
//      call, via the fault-injecting dlopen() interposer in
//      fault_inject_driver.c (see that file for how the substitution works
//      without shadowing the real driver for anything else).
//   3. Asserts via cuMemGetInfo that free device memory before and after the
//      forced-failure allocations is unchanged, i.e. nothing leaked.
//
// This requires the test binary to run with LD_PRELOAD listing, in order,
// the fault-injecting shim ahead of libvgpu.so:
//   LD_PRELOAD=<path to fault_inject_driver.so>:<path to libvgpu.so>
// LD_LIBRARY_PATH is left untouched -- the real driver loads normally, same
// as for every other test. See test/CMakeLists.txt (the
// alloc_async_failure_leak CTest entry) for how this is wired up
// automatically when the vgpu target is built.
//
// To confirm this test actually exercises the fix (and isn't vacuously
// passing): temporarily revert the free_unlisted_alloc change in
// src/allocator/allocator.c, rebuild, and re-run this test -- it must fail
// with "bytes not reclaimed (leak reproduced)".
//
// Run standalone (from a build directory with the vgpu target already built):
//   ctest -R alloc_async_failure_leak --output-on-failure

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

#define ALLOC_BYTES (32ull * 1024 * 1024) /* 32 MiB */
#define ITERATIONS 4

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

/* Print a CUresult, flagging codes the driver itself does not recognise. */
static void describe(const char *label, CUresult res) {
    const char *name = NULL;
    CUresult q = cuGetErrorName(res, &name);
    if (q != CUDA_SUCCESS || name == NULL)
        printf("  %-40s -> %d  <unrecognized error code>\n", label, (int)res);
    else
        printf("  %-40s -> %d  (%s)\n", label, (int)res, name);
}

int main(void) {
    CHECK_DRV(cuInit(0));

    CUdevice dev;
    CHECK_DRV(cuDeviceGet(&dev, TEST_DEVICE_ID));

    CUcontext ctx;
#if CUDA_VERSION >= 13000
    CHECK_DRV(cuCtxCreate(&ctx, NULL, 0, dev));
#else
    CHECK_DRV(cuCtxCreate(&ctx, 0, dev));
#endif

    CUstream stream;
    CHECK_DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    int failures = 0;
    size_t free0, free1, total;

    /* [A] Force the allocator's internal cuDeviceGetMemPool call (the one
     * add_chunk_async() makes right after cuMemAllocAsync succeeds) to fail
     * on the next call only, then drive the allocation through the real
     * hooked cuMemAllocAsync entry point. */
    printf("[A] cuMemAllocAsync succeeds, allocator's cuDeviceGetMemPool forced to fail\n");
    CHECK_DRV(cuMemGetInfo(&free0, &total));
    for (int i = 0; i < ITERATIONS; i++) {
        CUdeviceptr d = 0;
        setenv("HAMI_TEST_FAIL_NEXT_GETMEMPOOL", "1", 1);
        CUresult res = cuMemAllocAsync(&d, ALLOC_BYTES, stream);
        describe("cuMemAllocAsync (forced post-alloc failure)", res);
        if (res == CUDA_SUCCESS) {
            fprintf(stderr, "expected forced failure but call succeeded\n");
            failures++;
            /* Allocation did succeed after all: free it so it doesn't
             * pollute the leak check below. */
            cuMemFreeAsync(d, stream);
        }
        /* No cuMemFreeAsync here: the point of this test is that the
         * allocator itself must have freed the GPU memory when its internal
         * cuDeviceGetMemPool call failed. Freeing it ourselves would hide a
         * regression instead of catching it. */
    }
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuMemGetInfo(&free1, &total));
    printf("  free before=%zu after=%zu (iterations=%d)\n", free0, free1, ITERATIONS);
    if (free1 < free0) {
        fprintf(stderr, "FAIL: %zu bytes not reclaimed (leak reproduced)\n", free0 - free1);
        failures++;
    }

    /* [B] Force the allocator's internal cuMemPoolGetAttribute call (the one
     * account_async_chunk() makes, shared by add_chunk_async() and
     * add_chunk_from_pool_async()) to fail on the next call only, then drive
     * the allocation through the real hooked cuMemAllocFromPoolAsync entry
     * point. */
    printf("[B] cuMemAllocFromPoolAsync succeeds, allocator's cuMemPoolGetAttribute forced to fail\n");
    CUmemPoolProps props;
    memset(&props, 0, sizeof(props));
    props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.id = dev;
    CUmemoryPool pool;
    CHECK_DRV(cuMemPoolCreate(&pool, &props));

    CHECK_DRV(cuMemGetInfo(&free0, &total));
    for (int i = 0; i < ITERATIONS; i++) {
        CUdeviceptr d = 0;
        setenv("HAMI_TEST_FAIL_NEXT_POOLATTR", "1", 1);
        CUresult res = cuMemAllocFromPoolAsync(&d, ALLOC_BYTES, pool, stream);
        describe("cuMemAllocFromPoolAsync (forced post-alloc failure)", res);
        if (res == CUDA_SUCCESS) {
            fprintf(stderr, "expected forced failure but call succeeded\n");
            failures++;
            cuMemFreeAsync(d, stream);
        }
        /* Same reasoning as [A]: no cuMemFreeAsync here on purpose. */
    }
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuMemGetInfo(&free1, &total));
    printf("  free before=%zu after=%zu (iterations=%d)\n", free0, free1, ITERATIONS);
    if (free1 < free0) {
        fprintf(stderr, "FAIL: %zu bytes not reclaimed (leak reproduced)\n", free0 - free1);
        failures++;
    }
    cuMemPoolDestroy(pool);

    printf("\nResult: %s\n",
           failures ? "FAIL - post-allocation driver-call failure leaked memory (issue #2864)"
                    : "PASS - allocation reclaimed on forced post-allocation driver-call failure");

    cuStreamDestroy(stream);
    cuCtxDestroy(ctx);
    return failures ? 1 : 0;
}

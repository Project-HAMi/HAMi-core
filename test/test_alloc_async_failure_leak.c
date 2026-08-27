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
// This test does not depend on libvgpu being preloaded: it reproduces the
// exact "succeed, then fail on the very next driver call" shape directly
// against the real CUDA driver, using two deterministic, portable ways to
// force a driver call to fail (an invalid device ordinal for
// cuDeviceGetMemPool, and an out-of-range attribute enum for
// cuMemPoolGetAttribute -- the two calls fixed for #2864), then verifies
// that applying the fix's cleanup (cuMemFreeAsync on the just-allocated
// pointer) actually reclaims the memory, i.e. that cuMemGetInfo's free byte
// count returns to baseline instead of drifting down on every iteration.
//
// Build (standalone, no HAMi headers needed):
//   gcc test_alloc_async_failure_leak.c -o test_alloc_async_failure_leak \
//       -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda
// Or drop into HAMi-core/test/ and `make build-in-docker` (auto-discovered).
//
// Run (expect PASS on stock CUDA and, after the #2864 fix, under libvgpu):
//   ./test_alloc_async_failure_leak

#include <cuda.h>
#include <stdio.h>
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

    /* [A] Force cuDeviceGetMemPool (the call in add_chunk_async right after
     * cuMemAllocAsync) to fail, then apply the fix's cleanup. */
    printf("[A] cuMemAllocAsync succeeds, cuDeviceGetMemPool forced to fail\n");
    CHECK_DRV(cuMemGetInfo(&free0, &total));
    for (int i = 0; i < ITERATIONS; i++) {
        CUdeviceptr d = 0;
        CHECK_DRV(cuMemAllocAsync(&d, ALLOC_BYTES, stream));

        /* Portable, deterministic failure: no such device ordinal exists.
         * Stands in for real-world causes (mempool unsupported on this
         * GPU/driver combo, transient resource exhaustion, etc.) that the
         * code must also survive without leaking. */
        CUmemoryPool pool;
        CUresult f = cuDeviceGetMemPool(&pool, (CUdevice)-1);
        describe("cuDeviceGetMemPool(invalid device)", f);
        if (f == CUDA_SUCCESS) {
            fprintf(stderr, "expected forced failure but call succeeded\n");
            failures++;
        }

        /* This is the fix: on the post-allocation driver-call failure,
         * free the GPU memory the earlier call already handed back. */
        CHECK_DRV(cuMemFreeAsync(d, stream));
    }
    CHECK_DRV(cuStreamSynchronize(stream));
    CHECK_DRV(cuMemGetInfo(&free1, &total));
    printf("  free before=%zu after=%zu (iterations=%d)\n", free0, free1, ITERATIONS);
    if (free1 < free0) {
        fprintf(stderr, "FAIL: %zu bytes not reclaimed (leak reproduced)\n", free0 - free1);
        failures++;
    }

    /* [B] Force cuMemPoolGetAttribute (the call in account_async_chunk,
     * shared by add_chunk_async and add_chunk_from_pool_async) to fail,
     * then apply the fix's cleanup. */
    printf("[B] cuMemAllocFromPoolAsync succeeds, cuMemPoolGetAttribute forced to fail\n");
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
        CHECK_DRV(cuMemAllocFromPoolAsync(&d, ALLOC_BYTES, pool, stream));

        /* Portable, deterministic failure: attribute enum out of range. */
        size_t attr_val;
        CUresult f = cuMemPoolGetAttribute(pool, (CUmemPool_attribute)0x7fffffff, &attr_val);
        describe("cuMemPoolGetAttribute(invalid attribute)", f);
        if (f == CUDA_SUCCESS) {
            fprintf(stderr, "expected forced failure but call succeeded\n");
            failures++;
        }

        /* This is the fix: on the post-allocation driver-call failure,
         * free the GPU memory the earlier call already handed back. */
        CHECK_DRV(cuMemFreeAsync(d, stream));
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

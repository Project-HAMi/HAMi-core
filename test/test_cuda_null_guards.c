/* Regression test for NULL-pointer guards in CUDA hooks.
 *
 * NVIDIA OptiX/Aftermath internal init paths historically pass NULL into
 * cuMemAlloc_v2 during fallback probes. Without explicit guards in our
 * hooks the LD_PRELOAD-injected libvgpu.so would dereference NULL inside
 * allocate_raw and SegFault Isaac Sim Kit at startup. This test asserts
 * the hook returns a non-success error code and does not crash.
 *
 * Pattern matches commit 03f99d7 (cuMemGetInfo_v2 NULL forward).
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

extern CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);

static void test_cuMemAlloc_v2_null_dptr(void) {
    CUresult r = cuMemAlloc_v2(NULL, 4096);
    assert(r != CUDA_SUCCESS);
    printf("[OK] cuMemAlloc_v2(NULL, 4096) returned %d (non-zero, no crash)\n", r);
}

static void test_cuMemAlloc_v2_zero_size(void) {
    CUdeviceptr dptr = 0;
    CUresult r = cuMemAlloc_v2(&dptr, 0);
    printf("[OK] cuMemAlloc_v2(&dptr, 0) returned %d\n", r);
}

int main(void) {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d (skipping - no GPU?)\n", r);
        return 0;
    }
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate_v2(&ctx, 0, dev);

    test_cuMemAlloc_v2_null_dptr();
    test_cuMemAlloc_v2_zero_size();

    cuCtxDestroy_v2(ctx);
    return 0;
}

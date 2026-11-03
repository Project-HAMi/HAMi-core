// Copyright 2026 The HAMi Authors.
// Regression test for #390: VMM memory must stay charged while it is mapped,
// even after cuMemRelease, and be uncharged once it is also unmapped.
// Run with CUDA_DEVICE_MEMORY_LIMIT=1g.
#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define MIB (1024ULL * 1024ULL)
#define BIG (768 * MIB)

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, (int)_e);                                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define EXPECT(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static CUmemAllocationProp prop;

/* Returns 1 if a second BIG allocation fits under the limit, else 0. */
static int big_fits(void) {
    CUmemGenericAllocationHandle h;
    CUresult r = cuMemCreate(&h, BIG, &prop, 0);
    if (r == CUDA_SUCCESS) {
        cuMemRelease(h);
        return 1;
    }
    return 0;
}

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr va;
    CUmemGenericAllocationHandle h, h2;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    CHECK(cuMemAddressReserve(&va, BIG, 0, 0, 0));

    // Released while mapped: still charged until unmapped.
    CHECK(cuMemCreate(&h, BIG, &prop, 0));
    CHECK(cuMemMap(va, BIG, 0, h, 0));
    CHECK(cuMemRelease(h));
    EXPECT(!big_fits(), "release while mapped dropped the charge");
    CHECK(cuMemUnmap(va, BIG));
    EXPECT(big_fits(), "charge not dropped after release + unmap");

    // Unmapped first: still charged until released.
    CHECK(cuMemCreate(&h, BIG, &prop, 0));
    CHECK(cuMemMap(va, BIG, 0, h, 0));
    CHECK(cuMemUnmap(va, BIG));
    EXPECT(!big_fits(), "unmap before release dropped the charge");
    CHECK(cuMemRelease(h));
    EXPECT(big_fits(), "charge not dropped after unmap + release");

    // A retained handle keeps the memory alive after the original release.
    CHECK(cuMemCreate(&h, BIG, &prop, 0));
    CHECK(cuMemMap(va, BIG, 0, h, 0));
    CHECK(cuMemRetainAllocationHandle(&h2, (void *)va));
    CHECK(cuMemRelease(h));
    CHECK(cuMemUnmap(va, BIG));
    EXPECT(!big_fits(), "retained handle lost its charge");
    CHECK(cuMemRelease(h2));
    EXPECT(big_fits(), "charge not dropped after last release");

    // Never mapped: released memory is uncharged right away.
    CHECK(cuMemCreate(&h, BIG, &prop, 0));
    CHECK(cuMemRelease(h));
    EXPECT(big_fits(), "unmapped handle still charged after release");

    CHECK(cuMemAddressFree(va, BIG));
    CHECK(cuDevicePrimaryCtxRelease(dev));
    printf("PASS\n");
    return 0;
}

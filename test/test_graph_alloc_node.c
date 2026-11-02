// Regression test for #387: a graph memory allocation node larger than the
// device memory limit must be rejected, a small one must still be accepted.
// Run with CUDA_DEVICE_MEMORY_LIMIT=1g.
#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, (int)_e);                                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static CUresult add_node(CUgraph g, size_t bytes) {
    CUDA_MEM_ALLOC_NODE_PARAMS prm;
    CUgraphNode n;
    memset(&prm, 0, sizeof(prm));
    prm.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    prm.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prm.poolProps.location.id = 0;
    prm.bytesize = bytes;
    return cuGraphAddMemAllocNode(&n, g, NULL, 0, &prm);
}

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUgraph g;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    CHECK(cuGraphCreate(&g, 0));

    CHECK(add_node(g, 128ull << 20));
    CUresult r = add_node(g, 2048ull << 20);
    if (r != CUDA_ERROR_OUT_OF_MEMORY) {
        fprintf(stderr, "FAIL: 2 GiB node under 1 GiB limit -> %d\n", (int)r);
        return 1;
    }

    CHECK(cuGraphDestroy(g));

    // Two 768 MiB graphs together exceed the 1 GiB limit: the second is
    // rejected while the first is alive, accepted once it is destroyed.
    CUgraph g1, g2;
    CHECK(cuGraphCreate(&g1, 0));
    CHECK(cuGraphCreate(&g2, 0));
    CHECK(add_node(g1, 768ull << 20));
    r = add_node(g2, 768ull << 20);
    if (r != CUDA_ERROR_OUT_OF_MEMORY) {
        fprintf(stderr, "FAIL: second 768 MiB graph -> %d\n", (int)r);
        return 1;
    }
    CHECK(cuGraphDestroy(g1));
    CHECK(add_node(g2, 768ull << 20));
    printf("PASS\n");
    return 0;
}

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

static CUresult add_node(CUgraph g, size_t bytes, CUdeviceptr *ptr, int with_free) {
    CUDA_MEM_ALLOC_NODE_PARAMS prm;
    CUgraphNode n, f;
    memset(&prm, 0, sizeof(prm));
    prm.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    prm.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prm.poolProps.location.id = 0;
    prm.bytesize = bytes;
    CUresult r = cuGraphAddMemAllocNode(&n, g, NULL, 0, &prm);
    if (r != CUDA_SUCCESS) return r;
    if (ptr) *ptr = prm.dptr;
    return with_free ? cuGraphAddMemFreeNode(&f, g, &n, 1, prm.dptr) : CUDA_SUCCESS;
}

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUgraph g;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    CUdeviceptr p;
    CUstream st;
    CHECK(cuStreamCreate(&st, CU_STREAM_NON_BLOCKING));

    // The accounting hook is also what cuGetProcAddress hands out (this is how
    // the runtime API reaches the driver).
    typedef CUresult (*add_node_fn)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
                                    CUDA_MEM_ALLOC_NODE_PARAMS *);
    add_node_fn pfn = NULL;
    CUdriverProcAddressQueryResult qr;
    int drv;
    CHECK(cuDriverGetVersion(&drv));
    CHECK(cuGetProcAddress("cuGraphAddMemAllocNode", (void **)&pfn, drv, 0, &qr));
    CHECK(cuGraphCreate(&g, 0));
    {
        CUDA_MEM_ALLOC_NODE_PARAMS prm;
        CUgraphNode n;
        memset(&prm, 0, sizeof(prm));
        prm.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        prm.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prm.bytesize = 2048ull << 20;
        if (pfn == NULL || pfn(&n, g, NULL, 0, &prm) != CUDA_ERROR_OUT_OF_MEMORY) {
            fprintf(stderr, "FAIL: proc-address lookup bypasses the limit\n");
            return 1;
        }
    }
    CHECK(cuGraphDestroy(g));

    // A node larger than the limit is rejected, a small one is accepted.
    CHECK(cuGraphCreate(&g, 0));
    CHECK(add_node(g, 128ull << 20, NULL, 1));
    CUresult r = add_node(g, 2048ull << 20, NULL, 1);
    if (r != CUDA_ERROR_OUT_OF_MEMORY) {
        fprintf(stderr, "FAIL: 2 GiB node under 1 GiB limit -> %d\n", (int)r);
        return 1;
    }
    CHECK(cuGraphDestroy(g));

    // With a free node, memory is released when the graph is destroyed.
    CUgraph g1, g2;
    CHECK(cuGraphCreate(&g1, 0));
    CHECK(cuGraphCreate(&g2, 0));
    CHECK(add_node(g1, 768ull << 20, NULL, 1));
    r = add_node(g2, 768ull << 20, NULL, 1);
    if (r != CUDA_ERROR_OUT_OF_MEMORY) {
        fprintf(stderr, "FAIL: second 768 MiB graph -> %d\n", (int)r);
        return 1;
    }
    CHECK(cuGraphDestroy(g1));
    CHECK(add_node(g2, 768ull << 20, NULL, 1));
    CHECK(cuGraphDestroy(g2));

    // A graph exec keeps the memory after the graph is destroyed (PyTorch
    // destroys the graph right after instantiating it), until the exec goes.
    CUgraphExec ex;
    CUDA_GRAPH_INSTANTIATE_PARAMS ip;
    for (int with_params = 0; with_params < 2; with_params++) {
        CHECK(cuGraphCreate(&g1, 0));
        CHECK(add_node(g1, 768ull << 20, NULL, 1));
        if (with_params) {
            memset(&ip, 0, sizeof(ip));
            CHECK(cuGraphInstantiateWithParams(&ex, g1, &ip));
        } else {
            CHECK(cuGraphInstantiateWithFlags(&ex, g1, 0));
        }
        CHECK(cuGraphDestroy(g1));
        CHECK(cuGraphCreate(&g2, 0));
        r = add_node(g2, 768ull << 20, NULL, 1);
        if (r != CUDA_ERROR_OUT_OF_MEMORY) {
            fprintf(stderr, "FAIL: live exec not charged (params=%d) -> %d\n", with_params, (int)r);
            return 1;
        }
        CHECK(cuGraphExecDestroy(ex));
        CHECK(add_node(g2, 768ull << 20, NULL, 1));
        CHECK(cuGraphDestroy(g2));
    }

    // Without a free node the allocation outlives the graph, as in the
    // driver, and is released only when its pointer is freed.
    CHECK(cuGraphCreate(&g1, 0));
    CHECK(add_node(g1, 768ull << 20, &p, 0));
    CHECK(cuGraphInstantiateWithFlags(&ex, g1, 0));
    CHECK(cuGraphLaunch(ex, st));
    CHECK(cuStreamSynchronize(st));
    CHECK(cuGraphExecDestroy(ex));
    CHECK(cuGraphDestroy(g1));
    CHECK(cuGraphCreate(&g2, 0));
    r = add_node(g2, 768ull << 20, NULL, 1);
    if (r != CUDA_ERROR_OUT_OF_MEMORY) {
        fprintf(stderr, "FAIL: leaked graph alloc not charged -> %d\n", (int)r);
        return 1;
    }
    CHECK(cuMemFree(p));
    CHECK(add_node(g2, 768ull << 20, NULL, 1));
    printf("PASS\n");
    return 0;
}

// Copyright 2026 The HAMi Authors.
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

#define EXPECT_OOM(what, call)                                                \
    do {                                                                       \
        CUresult _r = (call);                                                  \
        if (_r != CUDA_ERROR_OUT_OF_MEMORY) {                                  \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, what, (int)_r); \
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

/** Check graph allocation quota rejection, rollback, and output correctness. */
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

    // Graphs share the device graph pool: two live graphs with free nodes cost
    // the larger one, not the sum (#395). 400 + 400 MiB fits in 768 MiB.
    CUgraph g1, g2;
    CUdeviceptr q;
    CHECK(cuGraphCreate(&g1, 0));
    CHECK(cuGraphCreate(&g2, 0));
    CHECK(add_node(g1, 400ull << 20, NULL, 1));
    CHECK(add_node(g2, 400ull << 20, NULL, 1));
    EXPECT_OOM("alloc over the shared pool", cuMemAlloc(&q, 500ull << 20));
    CHECK(cuMemAlloc(&q, 300ull << 20));  // 400 pool + 300 fits, 800 + 300 would not
    CHECK(cuMemFree(q));
    // ...but one graph is charged for all of its own allocations.
    EXPECT_OOM("second allocation of one graph", add_node(g1, 400ull << 20, NULL, 1));
    CHECK(cuGraphDestroy(g1));
    CHECK(cuGraphDestroy(g2));
    CHECK(cuMemAlloc(&q, 700ull << 20));  // the pool charge is gone
    CHECK(cuMemFree(q));

    // A graph exec keeps the memory after the graph is destroyed (PyTorch
    // destroys the graph right after instantiating it), until the exec goes.
    CUgraphExec ex;
    CUDA_GRAPH_INSTANTIATE_PARAMS ip;
    for (int with_params = 0; with_params < 2; with_params++) {
        CHECK(cuGraphCreate(&g1, 0));
        CHECK(add_node(g1, 600ull << 20, NULL, 1));
        if (with_params) {
            memset(&ip, 0, sizeof(ip));
            CHECK(cuGraphInstantiateWithParams(&ex, g1, &ip));
        } else {
            CHECK(cuGraphInstantiateWithFlags(&ex, g1, 0));
        }
        CHECK(cuGraphDestroy(g1));
        EXPECT_OOM("alloc while an exec is alive", cuMemAlloc(&q, 300ull << 20));
        CHECK(cuGraphExecDestroy(ex));
        CHECK(cuMemAlloc(&q, 300ull << 20));
        CHECK(cuMemFree(q));
    }

    // Without a free node the allocation outlives the graph, as in the
    // driver, and is released only when its pointer is freed.
    CHECK(cuGraphCreate(&g1, 0));
    CHECK(add_node(g1, 700ull << 20, &p, 0));
    CHECK(cuGraphInstantiateWithFlags(&ex, g1, 0));
    CHECK(cuGraphLaunch(ex, st));
    CHECK(cuStreamSynchronize(st));
    CHECK(cuGraphExecDestroy(ex));
    CHECK(cuGraphDestroy(g1));
    EXPECT_OOM("alloc while a launched graph alloc is leaked", cuMemAlloc(&q, 300ull << 20));
    CHECK(cuMemFree(p));
    CHECK(cuMemAlloc(&q, 300ull << 20));
    CHECK(cuMemFree(q));

    // A graph that never ran allocated nothing: destroying it frees the charge
    // even without a free node (#392), also after an exec was made but not run.
    for (int with_exec = 0; with_exec < 2; with_exec++) {
        CHECK(cuGraphCreate(&g1, 0));
        CHECK(add_node(g1, 700ull << 20, NULL, 0));
        if (with_exec) {
            CHECK(cuGraphInstantiateWithFlags(&ex, g1, 0));
            CHECK(cuGraphExecDestroy(ex));
        }
        EXPECT_OOM("alloc while an unrun graph is alive", cuMemAlloc(&q, 300ull << 20));
        CHECK(cuGraphDestroy(g1));
        CHECK(cuMemAlloc(&q, 700ull << 20));
        CHECK(cuMemFree(q));
    }

    // Instantiating with an upload allocates like a launch: still charged
    // after the graph and exec are gone (the driver keeps the memory, and it
    // cannot be freed with cuMemFree before a launch), so keep this last.
    CHECK(cuGraphCreate(&g1, 0));
    CHECK(add_node(g1, 700ull << 20, &p, 0));
    memset(&ip, 0, sizeof(ip));
    ip.flags = CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD;
    ip.hUploadStream = st;
    CHECK(cuGraphInstantiateWithParams(&ex, g1, &ip));
    CHECK(cuStreamSynchronize(st));
    CHECK(cuGraphExecDestroy(ex));
    CHECK(cuGraphDestroy(g1));
    EXPECT_OOM("alloc after an uploaded graph", cuMemAlloc(&q, 300ull << 20));
    printf("PASS\n");
    return 0;
}

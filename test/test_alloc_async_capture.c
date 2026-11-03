// Copyright 2026 The HAMi Authors.
// Regression test: cuMemAllocAsync inside a stream capture must not fail and
// must not invalidate the capture (cuMemPoolGetAttribute is illegal while
// capturing). Runs for global, thread-local and relaxed capture modes.
#include <cuda.h>
#include <stdio.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, (int)_e);                                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUstream s;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    CHECK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));

    CUstreamCaptureMode modes[] = {CU_STREAM_CAPTURE_MODE_GLOBAL,
                                   CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                                   CU_STREAM_CAPTURE_MODE_RELAXED};
    for (int i = 0; i < 3; i++) {
        CUdeviceptr p;
        CUgraph g;
        CHECK(cuStreamBeginCapture(s, modes[i]));
        CHECK(cuMemAllocAsync(&p, 32ull << 20, s));
        CHECK(cuMemFreeAsync(p, s));
        CHECK(cuStreamEndCapture(s, &g));
        CHECK(cuGraphDestroy(g));
    }

    // Capture allocations are charged to the graph: with a 1 GiB limit, two
    // live 768 MiB captures (each with a free node) cannot coexist, and destroying one frees room.
    CUdeviceptr p;
    CUgraph g1, g2;
    CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_RELAXED));
    CHECK(cuMemAllocAsync(&p, 768ull << 20, s));
    CHECK(cuMemFreeAsync(p, s));
    CHECK(cuStreamEndCapture(s, &g1));
    CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_RELAXED));
    if (cuMemAllocAsync(&p, 768ull << 20, s) != CUDA_ERROR_OUT_OF_MEMORY) {
        fprintf(stderr, "FAIL: second 768 MiB capture alloc not rejected\n");
        return 1;
    }
    cuStreamEndCapture(s, &g2);
    CHECK(cuGraphDestroy(g1));
    CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_RELAXED));
    CHECK(cuMemAllocAsync(&p, 768ull << 20, s));
    CHECK(cuMemFreeAsync(p, s));
    CHECK(cuStreamEndCapture(s, &g2));
    CHECK(cuGraphDestroy(g2));
    printf("PASS\n");
    return 0;
}

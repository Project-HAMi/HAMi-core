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

    // A capture that gets invalidated is discarded by the driver without a
    // cuGraphDestroy; its allocation never happened and must not stay charged (#392).
    CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
    CHECK(cuMemAllocAsync(&p, 768ull << 20, s));
    cuStreamQuery(s);  // not allowed while capturing: invalidates the capture
    if (cuStreamEndCapture(s, &g2) == CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: capture was not invalidated\n");
        return 1;
    }
    CUdeviceptr q;
    CHECK(cuMemAlloc(&q, 768ull << 20));
    CHECK(cuMemFree(q));

    // Same when the allocation happens on a stream that joined the capture:
    // the capture is terminated on the origin stream.
    CUstream side;
    CUevent ev;
    CHECK(cuStreamCreate(&side, CU_STREAM_NON_BLOCKING));
    CHECK(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
    CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
    CHECK(cuEventRecord(ev, s));
    CHECK(cuStreamWaitEvent(side, ev, 0));
    CHECK(cuMemAllocAsync(&p, 768ull << 20, side));
    cuStreamQuery(s);  // invalidates the capture
    cuEventRecord(ev, side);
    cuStreamWaitEvent(s, ev, 0);
    if (cuStreamEndCapture(s, &g2) == CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: joined-stream capture was not invalidated\n");
        return 1;
    }
    CHECK(cuMemAlloc(&q, 768ull << 20));
    CHECK(cuMemFree(q));
    printf("PASS\n");
    return 0;
}

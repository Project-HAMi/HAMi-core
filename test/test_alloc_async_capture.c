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
    printf("PASS\n");
    return 0;
}

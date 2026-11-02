// Host PID detection must not run in cuInit: it takes the node-wide postinit
// lock and creates primary contexts, which serialized hundreds of short lived
// children that only initialize the driver (#382). It runs on first GPU use.
// Run under LD_PRELOAD=libvgpu.so; exit 77 (skip) without it.
#include <cuda.h>
#include <stdio.h>

// Defined by libvgpu.so; null when it is not preloaded.
extern int pidfound __attribute__((weak));

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
    if (&pidfound == NULL) {
        fprintf(stderr, "SKIP: libvgpu.so not preloaded\n");
        return 77;
    }
    CUdevice dev;
    CUcontext ctx;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    if (pidfound != 0) {
        fprintf(stderr, "FAIL: host pid detected during cuInit\n");
        return 1;
    }
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    if (pidfound != 1) {
        fprintf(stderr, "FAIL: host pid not detected on first context\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}

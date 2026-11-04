// Copyright 2026 The HAMi Authors.
// Host PID detection must not run in cuInit: it takes the node-wide postinit
// lock and creates primary contexts, which serialized hundreds of short lived
// children that only initialize the driver (#382). It runs on first GPU use.
// Run under LD_PRELOAD=libvgpu.so; exit 77 (skip) without it.
#include <cuda.h>
#include <stdio.h>
#include <string.h>

// CUDA 13 headers do not declare the older context creation entry points.
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams,
                        unsigned int flags, CUdevice dev);

// Exported by libvgpu.so; null when it is not preloaded.
extern int vgpu_hostpid_found(void) __attribute__((weak));

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, (int)_e);                                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// usage: test_lazy_hostpid [primary|v2|v3|v4|green]: the call that creates the first context.
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "primary";
    if (vgpu_hostpid_found == NULL) {
        fprintf(stderr, "SKIP: libvgpu.so not preloaded\n");
        return 77;
    }
    CUdevice dev;
    CUcontext ctx;
    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "SKIP: no usable GPU\n");
        return 77;
    }
    CHECK(cuDeviceGet(&dev, 0));
    if (vgpu_hostpid_found() != 0) {
        fprintf(stderr, "FAIL: host pid detected during cuInit\n");
        return 1;
    }
    if (!strcmp(mode, "v2")) {
        CHECK(cuCtxCreate_v2(&ctx, 0, dev));
    } else if (!strcmp(mode, "v3")) {
        CHECK(cuCtxCreate_v3(&ctx, NULL, 0, 0, dev));
    } else if (!strcmp(mode, "v4")) {
        CUctxCreateParams params;
        memset(&params, 0, sizeof(params));
        CHECK(cuCtxCreate(&ctx, &params, 0, dev));
    } else if (!strcmp(mode, "green")) {
        CUdevResource resource;
        CUdevResourceDesc desc;
        CUgreenCtx green;
        // Not every device supports green contexts: that is a skip, not a failure.
        CUresult r = cuDeviceGetDevResource(dev, &resource, CU_DEV_RESOURCE_TYPE_SM);
        if (r == CUDA_SUCCESS) r = cuDevResourceGenerateDesc(&desc, &resource, 1);
        if (r == CUDA_SUCCESS) r = cuGreenCtxCreate(&green, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM);
        if (r == CUDA_ERROR_NOT_SUPPORTED) {
            fprintf(stderr, "SKIP: green contexts not supported\n");
            return 77;
        }
        CHECK(r);
    } else {
        CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    }
    if (vgpu_hostpid_found() != 1) {
        fprintf(stderr, "FAIL: host pid not detected on first context (%s)\n", mode);
        return 1;
    }
    printf("PASS\n");
    return 0;
}

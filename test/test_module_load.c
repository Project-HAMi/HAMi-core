/*
 * Module globals are allocated by the driver when a module is loaded, not
 * through cuMemAlloc. Run under LD_PRELOAD=libvgpu.so with a memory limit:
 * free memory must drop by the size of the global after the load and come
 * back after the unload.
 */
#include <cuda.h>
#include <stdio.h>

#include "test_utils.h"

#define GLOBAL_MIB 256

static const char ptx[] =
    ".version 6.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    ".visible .global .align 8 .b8 big[268435456];\n"
    ".visible .entry touch() {\n"
    "    .reg .u64 %rd<2>;\n"
    "    mov.u64 %rd1, big;\n"
    "    st.global.u8 [%rd1], 0;\n"
    "    ret;\n"
    "}\n";

static size_t free_mib(void) {
    size_t free = 0, total = 0;
    cuMemGetInfo(&free, &total);
    return free >> 20;
}

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUmodule module;
    size_t before, loaded, unloaded;

    CHECK_DRV_API(cuInit(0));
    CHECK_DRV_API(cuDeviceGet(&dev, TEST_DEVICE_ID));
    CHECK_DRV_API(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_DRV_API(cuCtxSetCurrent(ctx));

    before = free_mib();
    CHECK_DRV_API(cuModuleLoadData(&module, ptx));
    loaded = free_mib();
    CHECK_DRV_API(cuModuleUnload(module));
    unloaded = free_mib();
    printf("free MiB: before=%zu loaded=%zu unloaded=%zu\n",
           before, loaded, unloaded);

    if (before < loaded + GLOBAL_MIB) {
        fprintf(stderr, "module global not counted: free dropped %zu MiB\n",
                before - loaded);
        return 1;
    }
    if (unloaded + 2 < before) {
        fprintf(stderr, "module charge not released: %zu MiB still counted\n",
                before - unloaded);
        return 1;
    }
    CHECK_DRV_API(cuDevicePrimaryCtxRelease(dev));
    printf("test_module_load passed\n");
    return 0;
}

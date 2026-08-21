#include <stdio.h>
#include <cuda.h>

#include "test/test_utils.h"

// Regression test for the managed-memory state reported by the two CUDA
// pointer-attribute APIs.
//
// cuPointerGetAttribute and cuPointerGetAttributes must report the same
// CU_POINTER_ATTRIBUTE_IS_MANAGED value for the same pointer, and a
// cuMemAllocManaged allocation must be reported as managed.
//
// Run under the HAMi-core hooks, e.g.:
//   LD_PRELOAD=./libvgpu.so ./test/test_pointer_attribute_managed

int main() {
    CHECK_DRV_API(cuInit(0));

    CUdevice device;
    CHECK_DRV_API(cuDeviceGet(&device, TEST_DEVICE_ID));

    CUcontext ctx;
#if CUDA_VERSION >= 13000
    CHECK_DRV_API(cuCtxCreate(&ctx, NULL, 0, device));
#else
    CHECK_DRV_API(cuCtxCreate(&ctx, 0, device));
#endif

    CUdeviceptr dptr;
    CHECK_DRV_API(cuMemAllocManaged(&dptr, 1 << 20, CU_MEM_ATTACH_GLOBAL));

    int managed_single = -1;
    CHECK_DRV_API(cuPointerGetAttribute(&managed_single,
        CU_POINTER_ATTRIBUTE_IS_MANAGED, dptr));

    int managed_multi = -1;
    CUpointer_attribute attributes[1] = {CU_POINTER_ATTRIBUTE_IS_MANAGED};
    void *data[1] = {&managed_multi};
    CHECK_DRV_API(cuPointerGetAttributes(1, attributes, data, dptr));

    CUcontext popped_ctx;
    CHECK_DRV_API(cuCtxPopCurrent(&popped_ctx));

    int memory_type = -1;
    CUpointer_attribute error_attributes[1] = {
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE
    };
    void *error_data[1] = {&memory_type};
    CUresult error_result = cuPointerGetAttributes(
        1, error_attributes, error_data, dptr);

    CHECK_DRV_API(cuCtxPushCurrent(popped_ctx));

    if (error_result != CUDA_ERROR_INVALID_CONTEXT) {
        fprintf(stderr,
            "cuPointerGetAttributes returned %d without a current context, "
            "expected %d\n", error_result, CUDA_ERROR_INVALID_CONTEXT);
        return -1;
    }

    if (memory_type != -1) {
        fprintf(stderr,
            "cuPointerGetAttributes changed output after an error: %d\n",
            memory_type);
        return -1;
    }

    printf("IS_MANAGED: cuPointerGetAttribute=%d cuPointerGetAttributes=%d\n",
        managed_single, managed_multi);

    if (managed_single != managed_multi) {
        fprintf(stderr,
            "cuPointerGetAttribute and cuPointerGetAttributes disagree on "
            "IS_MANAGED: %d vs %d\n", managed_single, managed_multi);
        return -1;
    }

    if (managed_multi != 1) {
        fprintf(stderr,
            "managed allocation reported as not managed: IS_MANAGED=%d\n",
            managed_multi);
        return -1;
    }

    cuMemFree(dptr);
    CHECK_DRV_API(cuCtxDestroy(ctx));
    printf("test_pointer_attribute_managed passed\n");
    return 0;
}

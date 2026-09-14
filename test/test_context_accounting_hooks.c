#ifdef NDEBUG
#undef NDEBUG
#endif
#include <stdio.h>

#include "context_hook_fixture.h"

static void test_reset_preserves_retained_usage(void) {
    CUcontext ctx;
    const CUdevice dev = 0;

    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxReset_v2(dev) == CUDA_SUCCESS);
    assert(driver_refs[dev] == 2);
    assert(context_charge[dev] == 0);
    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(driver_refs[dev] == 2);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == 0);
}

static void test_failed_reset_removal_keeps_the_charge(void) {
    CUcontext ctx;
    const CUdevice dev = 2;

    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    fail_remove = 1;
    assert(cuDevicePrimaryCtxReset_v2(dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    fail_remove = 0;
    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == 0);
}

static void test_failed_release_removal_keeps_the_charge(void) {
    CUcontext ctx;
    const CUdevice dev = 1;

    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    fail_remove = 1;
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(driver_refs[dev] == 0);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    fail_remove = 0;
    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == 0);
}

static void test_driver_errors_do_not_change_accounting(void) {
    CUcontext ctx;
    const CUdevice dev = 3;

    driver_error = CUDA_ERROR_OUT_OF_MEMORY;
    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == driver_error);
    assert(context_charge[dev] == 0);
    driver_error = CUDA_SUCCESS;
    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    driver_error = CUDA_ERROR_INVALID_CONTEXT;
    assert(cuDevicePrimaryCtxReset_v2(dev) == driver_error);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == driver_error);
    assert(driver_refs[dev] == 1);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    driver_error = CUDA_SUCCESS;
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(context_charge[dev] == 0);
}

static void test_failed_charge_is_deferred(void) {
    CUcontext ctx;
    const CUdevice dev = 4;

    fail_add = 1;
    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    assert(driver_refs[dev] == 1);
    assert(context_charge[dev] == 0);
    fail_add = 0;
    assert(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
    assert(driver_refs[dev] == 2);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(driver_refs[dev] == 1);
    assert(context_charge[dev] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(dev) == CUDA_SUCCESS);
    assert(driver_refs[dev] == 0);
    assert(context_charge[dev] == 0);
}

static void test_out_of_range_device_skips_accounting(void) {
    CUcontext ctx;
    const CUdevice devs[] = {-1, CUDA_DEVICE_MAX_COUNT};
    size_t i;

    for (i = 0; i < sizeof(devs) / sizeof(devs[0]); i++) {
        out_of_range_driver_calls = 0;
        assert(cuDevicePrimaryCtxRetain(&ctx, devs[i]) == CUDA_SUCCESS);
        assert(cuDevicePrimaryCtxReset_v2(devs[i]) == CUDA_SUCCESS);
        assert(cuDevicePrimaryCtxRelease_v2(devs[i]) == CUDA_SUCCESS);
        assert(out_of_range_driver_calls == 3);
    }
}

int main(void) {
    test_reset_preserves_retained_usage();
    test_failed_reset_removal_keeps_the_charge();
    test_failed_release_removal_keeps_the_charge();
    test_driver_errors_do_not_change_accounting();
    test_failed_charge_is_deferred();
    test_out_of_range_device_skips_accounting();
    puts("context accounting hook tests passed");
    return 0;
}

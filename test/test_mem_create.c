#include <stdio.h>
#include <cuda.h>
#include <pthread.h>

#include "test_utils.h"

#define NUM_THREADS 4
#define ALLOC_SIZE  (64 * 1024 * 1024)

static CUdevice g_device;

void *thread_func(void *arg) {
    int tid = *(int *)arg;

    CUmemAllocationProp prop = {0};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = g_device;

    size_t granularity = 0;
    CUresult res = cuMemGetAllocationGranularity(&granularity, &prop,
                                                  CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "thread %d: cuMemGetAllocationGranularity failed: %d\n", tid, res);
        return NULL;
    }

    size_t size = ((ALLOC_SIZE + granularity - 1) / granularity) * granularity;

    CUmemGenericAllocationHandle handle;
    res = cuMemCreate(&handle, size, &prop, 0);
    printf("thread %d: cuMemCreate returned %d (size=%zu)\n", tid, res, size);

    if (res == CUDA_SUCCESS) {
        cuMemRelease(handle);
        return (void *)1;
    }

    return NULL;
}

int main() {
    CHECK_DRV_API(cuInit(0));

    CHECK_DRV_API(cuDeviceGet(&g_device, TEST_DEVICE_ID));

    CUcontext ctx;
#if CUDA_VERSION >= 13000
    CHECK_DRV_API(cuCtxCreate(&ctx, NULL, 0, g_device));
#else
    CHECK_DRV_API(cuCtxCreate(&ctx, 0, g_device));
#endif

    pthread_t threads[NUM_THREADS];
    int tids[NUM_THREADS];
    int i;

    for (i = 0; i < NUM_THREADS; i++) {
        tids[i] = i;
        pthread_create(&threads[i], NULL, thread_func, &tids[i]);
    }

    int success_count = 0;
    for (i = 0; i < NUM_THREADS; i++) {
        void *ret = NULL;
        pthread_join(threads[i], &ret);
        if (ret != NULL)
            success_count++;
    }

    printf("%d/%d threads succeeded\n", success_count, NUM_THREADS);

    if (success_count != NUM_THREADS) {
        fprintf(stderr, "expected all threads to succeed\n");
        return 1;
    }

    CHECK_DRV_API(cuCtxDestroy(ctx));
    return 0;
}

// Copyright 2026 The HAMi Authors.
/* GPU-free regression: link the production wrappers/allocator and replace only
 * their driver and accounting dependencies. No application-specific code. */
#include <stdio.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];
static size_t usage, limit;
static CUresult query_result, context_result, free_result;
static int query_calls, free_calls, accounting_calls, failures;
static CUdeviceptr freed;
static size_t deducted;
static int deducted_dev;

#define EXPECT(x) do { \
    if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); failures++; \
} } while (0)

void ensure_initialized(void) {}
unsigned int cuda_to_nvml_map(unsigned int dev) { return dev; }
uint64_t get_current_device_memory_usage(const int dev) { return usage; }
uint64_t get_current_device_memory_limit(const int dev) { return limit; }
CUresult cuCtxGetDevice(CUdevice *dev) { *dev = 0; return context_result; }
int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    EXPECT(pid == getpid()); EXPECT(type == 2);
    accounting_calls++; deducted = bytes; deducted_dev = dev; return 0;
}
static CUresult driver_query(size_t *available, size_t *total) {
    query_calls++;
    EXPECT(available != NULL && total != NULL);
    if (available) *available = 900;
    if (total) *total = 1000;
    return query_result;
}
static CUresult driver_free(CUdeviceptr ptr) {
    /* Production must release the allocator lock before entering the driver. */
    int rc = pthread_mutex_trylock(&mutex);
    EXPECT(rc == 0);
    if (rc == 0) pthread_mutex_unlock(&mutex);
    free_calls++; freed = ptr; return free_result;
}
static void test_query(void) {
    const size_t limits[] = {0, 600, 1500};
    const size_t usages[] = {0, 200, 600, 1000, 1200};
    for (int l = 0; l < 3; l++) for (int u = 0; u < 5; u++) {
        limit = limits[l]; usage = usages[u];
        size_t expected_total = l == 1 ? 600 : 1000;
        size_t expected_free = usage < expected_total ? expected_total - usage : 0;
        for (int mask = 0; mask < 4; mask++) {
            size_t available = 42, total = 43;
            query_calls = 0;
            EXPECT(cuMemGetInfo_v2(mask & 1 ? &available : NULL,
                                 mask & 2 ? &total : NULL) == CUDA_SUCCESS);
            EXPECT(query_calls == 1);
            EXPECT(available == (mask & 1 ? expected_free : 42));
            EXPECT(total == (mask & 2 ? expected_total : 43));
        }
    }
    for (int mask = 0; mask < 4; mask++) {
        size_t available = 42, total = 43;
        query_result = CUDA_ERROR_INVALID_VALUE;
        EXPECT(cuMemGetInfo_v2(mask & 1 ? &available : NULL,
                             mask & 2 ? &total : NULL) == query_result);
        EXPECT(available == 42 && total == 43);
    }
    query_result = CUDA_SUCCESS;
    context_result = CUDA_ERROR_INVALID_CONTEXT;
    query_calls = 0;
    size_t available = 42, total = 43;
    EXPECT(cuMemGetInfo_v2(&available, &total) == context_result);
    EXPECT(query_calls == 0 && available == 42 && total == 43);
    context_result = CUDA_SUCCESS;
}
static void test_free(void) {
    allocated_list list = {0};
    device_overallocated = &list;
    for (int populated = 0; populated < 2; populated++) {
        if (populated) {
            allocated_list_entry *e = calloc(1, sizeof(*e));
            e->entry = calloc(1, sizeof(*e->entry));
            e->entry->address = 123; e->entry->length = 64; e->entry->dev = 7;
            list.head = list.tail = e; list.length = 1;
        }
        for (int error = 0; error < 2; error++) {
            free_result = error ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
            free_calls = 0;
            EXPECT(cuMemFree_v2(456) == free_result);
            EXPECT(free_calls == 1 && freed == 456);
            EXPECT(accounting_calls == 0 && list.length == (size_t)populated);
            if (populated) EXPECT(list.head == list.tail && list.head->entry->address == 123);
        }
    }
    free_result = CUDA_SUCCESS; free_calls = 0;
    EXPECT(cuMemFree_v2(123) == CUDA_SUCCESS);
    EXPECT(free_calls == 1 && freed == 123);
    EXPECT(accounting_calls == 1 && deducted == 64 && deducted_dev == 7);
    EXPECT(list.length == 0 && list.head == NULL && list.tail == NULL);
    free_calls = 0;
    EXPECT(cuMemFree_v2(0) == CUDA_SUCCESS && free_calls == 0);
    device_overallocated = NULL;
}
int main(void) {
    log_utils_init();
    cuda_library_entry[OVERRIDE_cuMemGetInfo_v2].fn_ptr = driver_query;
    cuda_library_entry[OVERRIDE_cuMemFree_v2].fn_ptr = driver_free;
    test_query(); test_free();
    printf("CUDA memory compatibility: %d failures\n", failures);
    return failures != 0;
}

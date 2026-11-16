// Copyright 2026 The HAMi Authors.
/* Link production allocation hooks and inject failures before native mutation. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define MIB (1ull << 20)
#define REQUIRE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); \
        exit(1); \
    } \
} while (0)

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];
static size_t used = 256 * MIB;
static int nodes, native_calls, undo_calls, fail_calloc, compete;
static CUresult native_result, competing_result;
static uintptr_t serial = 10;

void *__real_calloc(size_t count, size_t size);

/** Fail one bookkeeping allocation without changing the native graph. */
void *__wrap_calloc(size_t count, size_t size) {
    if (fail_calloc) {
        fail_calloc = 0;
        return NULL;
    }
    return __real_calloc(count, size);
}

void ensure_initialized(void) {}
void ensure_hostpid(void) {}
void lock_shrreg(void) {}
void unlock_shrreg(void) {}
unsigned int cuda_to_nvml_map(unsigned int dev) { return dev; }
uint64_t get_current_device_memory_limit(const int dev) { REQUIRE(dev == 0); return 1024 * MIB; }
size_t get_gpu_memory_usage(const int dev) { REQUIRE(dev == 0); return used; }
int clear_proc_slot_nolock(int force) { (void)force; return 0; }
CUresult cuCtxGetDevice(CUdevice *dev) { *dev = 0; return CUDA_SUCCESS; }
int add_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid; (void)type; REQUIRE(dev == 0); used += bytes; return 0;
}
/** Release a simulated charge without allowing an accounting underflow. */
int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid; (void)type; REQUIRE(dev == 0 && used >= bytes); used -= bytes; return 0;
}

/** Model another allocation arriving between admission and native completion. */
static CUresult native_allocate(CUdeviceptr *ptr, size_t size, CUstream stream) {
    (void)size; (void)stream; native_calls++;
    if (compete) competing_result = reserve_device_memory(0, 400 * MIB);
    if (native_result != CUDA_SUCCESS) return native_result;
    *ptr = ++serial * MIB; nodes++;
    return CUDA_SUCCESS;
}

/** Model an allocation node whose creation cannot be undone individually. */
static CUresult native_add(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                            size_t count, CUDA_MEM_ALLOC_NODE_PARAMS *params) {
    (void)graph; (void)deps; (void)count;
    CUresult result = native_allocate(&params->dptr, params->bytesize, NULL);
    if (result == CUDA_SUCCESS) *node = (CUgraphNode)++serial;
    return result;
}

static CUresult native_undo(CUgraphNode node) {
    (void)node; undo_calls++; return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult native_capture(CUstream stream, CUstreamCaptureStatus *status, cuuint64_t *id,
                                CUgraph *graph, const CUgraphNode **deps, size_t *count) {
    (void)stream; (void)deps; (void)count;
    *status = CU_STREAM_CAPTURE_STATUS_ACTIVE; *id = 123; *graph = (CUgraph)2;
    return CUDA_SUCCESS;
}

/** Add an allocation node through the production hook. */
static CUresult allocate(size_t size, CUdeviceptr *ptr) {
    CUDA_MEM_ALLOC_NODE_PARAMS params = {0};
    CUgraphNode node;
    params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    params.poolProps.location.id = 0;
    params.bytesize = size;
    CUresult result = cuGraphAddMemAllocNode(&node, (CUgraph)2, NULL, 0, &params);
    if (result == CUDA_SUCCESS) *ptr = params.dptr;
    return result;
}

/** Check quota rejection, native failure, heap failure, and concurrent admission. */
static void check_admission(void) {
    CUdeviceptr ptr = 99;
    native_calls = nodes = undo_calls = 0;
    REQUIRE(allocate(900 * MIB, &ptr) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(ptr == 99 && native_calls == 0 && nodes == 0 && used == 256 * MIB);

    fail_calloc = 1;
    REQUIRE(allocate(400 * MIB, &ptr) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(native_calls == 0 && nodes == 0 && undo_calls == 0 && used == 256 * MIB);

    native_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(allocate(400 * MIB, &ptr) == native_result);
    REQUIRE(native_calls == 1 && nodes == 0 && used == 256 * MIB);
    native_result = CUDA_SUCCESS;

    compete = 1;
    REQUIRE(allocate(500 * MIB, &ptr) == CUDA_SUCCESS);
    REQUIRE(competing_result == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(nodes == 1 && used == 756 * MIB);
    compete = 0;
    graph_mark_free((CUgraph)2, ptr);
    graph_release((CUgraph)2);
    REQUIRE(used == 256 * MIB && undo_calls == 0);

    graph_capture_begin((CUstream)3);
    REQUIRE(allocate(400 * MIB, &ptr) == CUDA_SUCCESS && used == 656 * MIB);
    cuuint64_t id = graph_capture_id((CUstream)3, 1);
    REQUIRE(id == 123);
    graph_capture_end(id, NULL);
    REQUIRE(used == 256 * MIB);
}

/** Run admission regressions for explicit allocation nodes. */
int main(void) {
    log_utils_init();
    cuda_library_entry[OVERRIDE_cuGraphAddMemAllocNode].fn_ptr = native_add;
    cuda_library_entry[OVERRIDE_cuGraphDestroyNode].fn_ptr = native_undo;
    cuda_library_entry[OVERRIDE_cuStreamGetCaptureInfo_v2].fn_ptr = native_capture;
    check_admission();
    puts("PASS: graph allocation admission, rollback, and concurrent reservation");
    return 0;
}

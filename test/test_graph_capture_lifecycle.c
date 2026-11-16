// Copyright 2026 The HAMi Authors.
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"

#define MIB (1ull << 20)
#define REQUIRE(x) do { \
    if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];
static size_t used = 256 * MIB;
static pthread_mutex_t shared = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static _Thread_local unsigned int current_context = 1;
static _Thread_local CUgraph capture_graphs[3];
static _Thread_local cuuint64_t capture_ids[3];
#define capture capture_graphs[current_context]
#define capture_id capture_ids[current_context]
static _Atomic unsigned int serial = 10;
static CUresult begin_result, end_result, info_result, context_result;
static int end_calls;
static int keep_capture, ready, finish, allocating, release_allocation, requested, begin_calls;
static allocated_list async_list;

void ensure_initialized(void) {}
void ensure_hostpid(void) {}
void lock_shrreg(void) { pthread_mutex_lock(&shared); }
void unlock_shrreg(void) { pthread_mutex_unlock(&shared); }
unsigned int cuda_to_nvml_map(unsigned int dev) { return dev; }
uint64_t get_current_device_memory_limit(const int dev) { (void)dev; return 1024 * MIB; }
size_t get_gpu_memory_usage(const int dev) { (void)dev; return used; }
int clear_proc_slot_nolock(int force) { (void)force; return 0; }
int add_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid; (void)dev; (void)type; used += bytes; return 0;
}
/** Refund simulated usage without allowing an underflow. */
int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid; (void)dev; (void)type; REQUIRE(used >= bytes); used -= bytes; return 0;
}
CUresult cuCtxGetDevice(CUdevice *dev) { *dev = 0; return CUDA_SUCCESS; }

/** Report capture state or the configured native query failure. */
static CUresult native_info(CUstream stream, CUstreamCaptureStatus *status, cuuint64_t *id,
                            CUgraph *graph, const CUgraphNode **deps,
                            const CUgraphEdgeData **data, size_t *count) {
    (void)stream; (void)deps; (void)data; (void)count;
    if (info_result != CUDA_SUCCESS) return info_result;
    *status = capture ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
    *graph = capture; *id = capture_id;
    return CUDA_SUCCESS;
}

/** Create an independent capture on each host thread. */
static CUresult native_begin(CUstream stream, CUstreamCaptureMode mode) {
    (void)stream; (void)mode;
    pthread_mutex_lock(&gate);
    begin_calls++;
    pthread_cond_broadcast(&changed);
    pthread_mutex_unlock(&gate);
    if (begin_result != CUDA_SUCCESS) return begin_result;
    capture_id = atomic_fetch_add(&serial, 1);
    capture = (CUgraph)(uintptr_t)capture_id;
    return CUDA_SUCCESS;
}

static CUresult native_begin_original(CUstream stream) {
    return native_begin(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
}

/** Simulate terminal and nonterminal native capture results. */
static CUresult native_end(CUstream stream, CUgraph *graph) {
    (void)stream; end_calls++;
    *graph = end_result == CUDA_SUCCESS ? capture : NULL;
    if (!keep_capture) capture = NULL;
    return end_result;
}

/** Return the active context or an injected native query failure. */
static CUresult native_current(CUcontext *ctx) {
    if (context_result != CUDA_SUCCESS) return context_result;
    *ctx = (CUcontext)(uintptr_t)current_context; return CUDA_SUCCESS;
}

static CUresult native_destroy(CUgraph graph) { (void)graph; return CUDA_SUCCESS; }

/** Keep quota and capture identity after a nonterminal native EndCapture error. */
static void check_failed_end(void) {
    CUgraph graph;
    cuuint64_t id;
    begin_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(cuStreamBeginCapture_v2((CUstream)10, CU_STREAM_CAPTURE_MODE_GLOBAL) == begin_result);
    REQUIRE(graph_capture_token((CUstream)10, &id) == 0);
    begin_result = CUDA_SUCCESS;
    REQUIRE(cuStreamBeginCapture_v2((CUstream)10, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    REQUIRE(graph_record(capture, 0, 100 * MIB, 1, capture_id) == CUDA_SUCCESS);
    uint64_t token = graph_capture_token((CUstream)10, &id);
    keep_capture = 1; end_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(cuStreamEndCapture((CUstream)10, &graph) == end_result);
    REQUIRE(graph_capture_token((CUstream)10, &id) == token && used == 356 * MIB);
    info_result = CUDA_ERROR_UNKNOWN;
    REQUIRE(cuStreamEndCapture((CUstream)10, &graph) == end_result);
    REQUIRE(graph_capture_token((CUstream)10, &id) == token && used == 356 * MIB);
    keep_capture = 0; end_result = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    REQUIRE(cuStreamEndCapture((CUstream)10, &graph) == end_result);
    REQUIRE(graph_capture_token((CUstream)10, &id) == 0 && used == 256 * MIB);
    info_result = end_result = CUDA_SUCCESS;
    REQUIRE(cuStreamBeginCapture_v2((CUstream)10, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    REQUIRE(graph_record(capture, 0, 100 * MIB, 1, capture_id) == CUDA_SUCCESS);
    end_result = CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
    REQUIRE(cuStreamEndCapture((CUstream)10, &graph) == end_result);
    REQUIRE(graph_capture_token((CUstream)10, &id) == 0 && used == 256 * MIB);
    end_result = CUDA_SUCCESS;
}

/** Hold another per-thread capture until the origin thread has ended its own. */
static void *other_capture(void *unused) {
    (void)unused;
    REQUIRE(cuStreamBeginCapture_v2(CU_STREAM_PER_THREAD, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == CUDA_SUCCESS);
    REQUIRE(graph_record(capture, 0, 200 * MIB, 2, capture_id) == CUDA_SUCCESS);
    pthread_mutex_lock(&gate);
    ready = 1; pthread_cond_broadcast(&changed);
    while (!finish) pthread_cond_wait(&changed, &gate);
    pthread_mutex_unlock(&gate);
    CUgraph graph;
    REQUIRE(cuStreamEndCapture(CU_STREAM_PER_THREAD, &graph) == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
    return NULL;
}

/** Isolate captures that use the same per-thread stream sentinel on different threads. */
static void check_thread_identity(void) {
    REQUIRE(cuStreamBeginCapture_v2(CU_STREAM_PER_THREAD, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == CUDA_SUCCESS);
    REQUIRE(graph_record(capture, 0, 100 * MIB, 1, capture_id) == CUDA_SUCCESS);
    pthread_t thread;
    REQUIRE(pthread_create(&thread, NULL, other_capture, NULL) == 0);
    pthread_mutex_lock(&gate);
    while (!ready) pthread_cond_wait(&changed, &gate);
    pthread_mutex_unlock(&gate);
    CUgraph graph;
    end_result = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    REQUIRE(cuStreamEndCapture(CU_STREAM_PER_THREAD, &graph) == end_result);
    REQUIRE(used == 456 * MIB);
    pthread_mutex_lock(&gate);
    finish = 1; pthread_cond_broadcast(&changed);
    pthread_mutex_unlock(&gate);
    REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(used == 256 * MIB);
    end_result = CUDA_SUCCESS;
}

/** Keep independent per-thread captures isolated across contexts on one host thread. */
static void check_context_identity(void) {
    current_context = 1;
    REQUIRE(cuStreamBeginCapture_v2(CU_STREAM_PER_THREAD, CU_STREAM_CAPTURE_MODE_RELAXED) == CUDA_SUCCESS);
    REQUIRE(graph_record(capture, 0, 100 * MIB, 1, capture_id) == CUDA_SUCCESS);
    current_context = 2;
    REQUIRE(cuStreamBeginCapture_v2(CU_STREAM_PER_THREAD, CU_STREAM_CAPTURE_MODE_RELAXED) == CUDA_SUCCESS);
    REQUIRE(graph_record(capture, 0, 200 * MIB, 2, capture_id) == CUDA_SUCCESS);
    cuuint64_t id;
    uint64_t other = graph_capture_token(CU_STREAM_PER_THREAD, &id);
    current_context = 1;
    CUgraph graph;
    end_result = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    REQUIRE(cuStreamEndCapture_ptsz(NULL, &graph) == end_result);
    REQUIRE(used == 456 * MIB);
    current_context = 2;
    REQUIRE(graph_capture_token(CU_STREAM_PER_THREAD, &id) == other);
    REQUIRE(cuStreamEndCapture_ptsz(NULL, &graph) == end_result);
    REQUIRE(used == 256 * MIB);
    current_context = 1;
    end_result = CUDA_SUCCESS;
}

/** Leave native capture active when its per-thread context identity cannot be resolved. */
static void check_context_failure(void) {
    REQUIRE(cuStreamBeginCapture_v2(CU_STREAM_PER_THREAD, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    REQUIRE(graph_record(capture, 0, 100 * MIB, 1, capture_id) == CUDA_SUCCESS);
    int before = end_calls;
    context_result = CUDA_ERROR_UNKNOWN;
    CUgraph graph;
    REQUIRE(cuStreamEndCapture_ptsz(NULL, &graph) == context_result);
    REQUIRE(end_calls == before && capture != NULL && used == 356 * MIB);
    context_result = CUDA_SUCCESS;
    end_result = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    REQUIRE(cuStreamEndCapture_ptsz(NULL, &graph) == end_result);
    REQUIRE(used == 256 * MIB);
    end_result = CUDA_SUCCESS;
}

/** Pause an actual allocator call immediately before native allocation. */
static CUresult native_allocate(CUdeviceptr *ptr, size_t bytes, CUstream stream) {
    (void)bytes; (void)stream;
    pthread_mutex_lock(&gate);
    allocating = 1; pthread_cond_broadcast(&changed);
    while (!release_allocation) pthread_cond_wait(&changed, &gate);
    pthread_mutex_unlock(&gate);
    *ptr = 3; return CUDA_SUCCESS;
}
static CUresult native_pool(CUmemoryPool *pool, CUdevice dev) {
    (void)dev; *pool = (CUmemoryPool)1; return CUDA_SUCCESS;
}
static CUresult native_attribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    (void)pool; (void)attr; *(cuuint64_t *)value = 0; return CUDA_SUCCESS;
}
static void *allocate_thread(void *unused) {
    (void)unused; CUdeviceptr ptr;
    REQUIRE(allocate_async_raw(&ptr, 16, (CUstream)10) == CUDA_SUCCESS); return NULL;
}
static void *begin_thread(void *unused) {
    (void)unused;
    pthread_mutex_lock(&gate);
    requested = 1; pthread_cond_broadcast(&changed);
    pthread_mutex_unlock(&gate);
    CUgraph graph;
    REQUIRE(cuStreamBeginCapture_v2((CUstream)10, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    REQUIRE(cuStreamEndCapture((CUstream)10, &graph) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    return NULL;
}

/** Prevent native capture entry between allocation classification and native completion. */
static void check_transition_serialization(void) {
    pthread_t allocator, capturer;
    REQUIRE(pthread_create(&allocator, NULL, allocate_thread, NULL) == 0);
    pthread_mutex_lock(&gate);
    while (!allocating) pthread_cond_wait(&changed, &gate);
    int before = begin_calls;
    pthread_mutex_unlock(&gate);
    REQUIRE(pthread_create(&capturer, NULL, begin_thread, NULL) == 0);
    pthread_mutex_lock(&gate);
    while (!requested) pthread_cond_wait(&changed, &gate);
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline); deadline.tv_sec++;
    while (begin_calls == before && pthread_cond_timedwait(&changed, &gate, &deadline) == 0) {}
    REQUIRE(begin_calls == before);
    release_allocation = 1; pthread_cond_broadcast(&changed);
    pthread_mutex_unlock(&gate);
    REQUIRE(pthread_join(allocator, NULL) == 0 && pthread_join(capturer, NULL) == 0);
    REQUIRE(begin_calls == before + 1);
    free(async_list.head->entry); free(async_list.head); LIST_INIT((&async_list));
}

/** Exercise production capture wrappers and allocator transition ordering. */
int main(void) {
    device_allocasync = &async_list;
    cuda_library_entry[OVERRIDE_cuStreamGetCaptureInfo_v3].fn_ptr = native_info;
    cuda_library_entry[OVERRIDE_cuCtxGetCurrent].fn_ptr = native_current;
    cuda_library_entry[OVERRIDE_cuStreamBeginCapture_v2].fn_ptr = native_begin;
    cuda_library_entry[OVERRIDE_cuStreamEndCapture].fn_ptr = native_end;
    cuda_library_entry[OVERRIDE_cuGraphDestroy].fn_ptr = native_destroy;
    cuda_library_entry[OVERRIDE_cuMemAllocAsync].fn_ptr = native_allocate;
    cuda_library_entry[OVERRIDE_cuDeviceGetMemPool].fn_ptr = native_pool;
    cuda_library_entry[OVERRIDE_cuMemPoolGetAttribute].fn_ptr = native_attribute;
    cuda_library_entry[OVERRIDE_cuStreamBeginCapture].fn_ptr = native_begin_original;
    CUgraph graph;
    REQUIRE(cuStreamBeginCapture((CUstream)10) == CUDA_SUCCESS);
    REQUIRE(cuStreamEndCapture((CUstream)10, &graph) == CUDA_SUCCESS);
    REQUIRE(cuStreamBeginCapture_ptsz(NULL) == CUDA_SUCCESS);
    REQUIRE(cuStreamEndCapture_ptsz(NULL, &graph) == CUDA_SUCCESS);
    check_failed_end(); check_thread_identity(); check_context_identity();
    check_context_failure(); check_transition_serialization();
    puts("PASS: capture identity, terminal errors, and allocation transition ordering");
    return 0;
}

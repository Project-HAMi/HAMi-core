// Copyright 2026 The HAMi Authors.
/* Exercise production accounting with an asynchronous fake driver. */
#include <stdint.h>
#include <stdatomic.h>
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
static size_t used = 256 * MIB, reserved;
static uintptr_t serial = 10;
static int nodes, submissions, instantiations, gpu_done;
static CUresult add_result, submit_result, event_result, pool_result, record_result;
static _Thread_local CUgraph capturing;
static _Thread_local cuuint64_t capture_id = 123;
static _Thread_local CUstreamCaptureMode capturing_mode;
static _Atomic int global_captures;
static CUcontext current_ctx = (CUcontext)1, pushed_ctx;
static struct { CUcontext ctx; cuuint64_t ctx_id; int alive; } events[8192];
static cuuint64_t context_ids[] = {0, 1, 2};
static int event_queries, pool_queries, events_created;
static pthread_mutex_t submit_gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t submit_cond = PTHREAD_COND_INITIALIZER;
static int block_submit, submit_entered;
static int block_sync, sync_entered;
static CUresult capture_end_result;
static CUresult capture_begin_result;
static CUresult capture_info_result;
static int keep_capture, block_capture_begin, block_capture_end, capture_entered;
static int capture_ready, capture_finish;

void ensure_initialized(void) {}
void lock_shrreg(void) {}
void unlock_shrreg(void) {}
unsigned int cuda_to_nvml_map(unsigned int dev) { return dev; }
uint64_t get_current_device_memory_limit(const int dev) { REQUIRE(dev == 0); return 1024 * MIB; }
size_t get_gpu_memory_usage(const int dev) { REQUIRE(dev == 0); return used; }
int clear_proc_slot_nolock(int force) { (void)force; return 0; }
int add_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid; (void)type; REQUIRE(dev == 0); used += bytes; return 0;
}
int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid; (void)type; REQUIRE(dev == 0 && used >= bytes); used -= bytes; return 0;
}
static CUresult driver_device(CUdevice *dev) { *dev = 0; return CUDA_SUCCESS; }

static CUresult driver_add(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                            size_t ndeps, CUDA_MEM_ALLOC_NODE_PARAMS *params) {
    (void)graph; (void)deps; (void)ndeps;
    if (add_result != CUDA_SUCCESS) return add_result;
    nodes++;
    *node = (CUgraphNode)++serial;
    params->dptr = ++serial * MIB;
    return CUDA_SUCCESS;
}
static CUresult driver_free_node(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                                  size_t ndeps, CUdeviceptr ptr) {
    (void)graph; (void)deps; (void)ndeps; (void)ptr;
    *node = (CUgraphNode)++serial;
    return CUDA_SUCCESS;
}
static CUresult driver_instantiate(CUgraphExec *exec, CUgraph graph, cuuint64_t flags) {
    (void)graph; (void)flags; instantiations++;
    *exec = (CUgraphExec)++serial;
    return CUDA_SUCCESS;
}
static CUresult driver_params(CUgraphExec *exec, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS *params) {
    (void)params;
    return driver_instantiate(exec, graph, 0);
}
static CUresult driver_submit(CUgraphExec exec, CUstream stream) {
    (void)exec; (void)stream; submissions++;
    if (block_submit) {
        pthread_mutex_lock(&submit_gate);
        submit_entered = 1;
        pthread_cond_broadcast(&submit_cond);
        while (block_submit) pthread_cond_wait(&submit_cond, &submit_gate);
        pthread_mutex_unlock(&submit_gate);
    }
    return submit_result;
}
static CUresult driver_context_sync(void) {
    pthread_mutex_lock(&submit_gate);
    sync_entered = 1;
    pthread_cond_broadcast(&submit_cond);
    while (block_sync) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
    return CUDA_SUCCESS;
}
static CUresult driver_stream_sync(CUstream stream) {
    (void)stream;
    return driver_context_sync();
}
static CUresult driver_destroy(void *handle) { (void)handle; return CUDA_SUCCESS; }
static CUresult driver_event_create(CUevent *event, unsigned int flags) {
    REQUIRE(flags == CU_EVENT_DISABLE_TIMING);
    if (event_result != CUDA_SUCCESS) return event_result;
    REQUIRE(current_ctx != NULL && serial + 1 < 8192);
    *event = (CUevent)++serial;
    events[serial].ctx = current_ctx;
    events[serial].ctx_id = context_ids[(uintptr_t)current_ctx];
    events[serial].alive = 1;
    events_created++;
    return CUDA_SUCCESS;
}
static CUcontext stream_ctx(CUstream stream) {
    if (stream == NULL || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD) return current_ctx;
    return stream == (CUstream)30 ? (CUcontext)2 : (CUcontext)1;
}
static CUresult driver_event_record(CUevent event, CUstream stream) {
    REQUIRE((uintptr_t)event < 8192);
    if (!events[(uintptr_t)event].alive || events[(uintptr_t)event].ctx != stream_ctx(stream))
        return CUDA_ERROR_INVALID_HANDLE;
    return record_result;
}
static CUresult driver_event_query(CUevent event) {
    REQUIRE(capturing == NULL && atomic_load(&global_captures) == 0);
    event_queries++;
    REQUIRE((uintptr_t)event < 8192);
    if (!events[(uintptr_t)event].alive) return CUDA_ERROR_INVALID_HANDLE;
    return gpu_done ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
static CUresult driver_event_destroy(CUevent event) {
    REQUIRE((uintptr_t)event < 8192 && events[(uintptr_t)event].alive);
    events[(uintptr_t)event].alive = 0;
    return CUDA_SUCCESS;
}
static CUresult driver_stream_ctx(CUstream stream, CUcontext *ctx) {
    *ctx = stream_ctx(stream); return CUDA_SUCCESS;
}
static CUresult driver_current(CUcontext *ctx) { *ctx = current_ctx; return CUDA_SUCCESS; }
static CUresult driver_context_id(CUcontext ctx, cuuint64_t *id) {
    REQUIRE((uintptr_t)ctx < 3);
    *id = context_ids[(uintptr_t)ctx]; return CUDA_SUCCESS;
}
static CUresult driver_push(CUcontext ctx) {
    pushed_ctx = current_ctx; current_ctx = ctx; return CUDA_SUCCESS;
}
static CUresult driver_pop(CUcontext *ctx) {
    *ctx = current_ctx; current_ctx = pushed_ctx; return CUDA_SUCCESS;
}
static CUresult driver_stream_id(CUstream stream, cuuint64_t *id) {
    *id = (uintptr_t)stream + 100; return CUDA_SUCCESS;
}
static CUresult driver_reserved(CUdevice dev, CUgraphMem_attribute attr, void *value) {
    REQUIRE(capturing == NULL);
    pool_queries++;
    REQUIRE(dev == 0 && attr == CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT);
    if (pool_result != CUDA_SUCCESS) return pool_result;
    *(cuuint64_t *)value = reserved; return CUDA_SUCCESS;
}
static CUresult driver_trim(CUdevice dev) {
    REQUIRE(capturing == NULL);
    REQUIRE(dev == 0);
    if (gpu_done) reserved = 0;
    return CUDA_SUCCESS;
}
static CUresult driver_capture(CUstream stream, CUstreamCaptureStatus *status, cuuint64_t *id,
                                CUgraph *graph, const CUgraphNode **deps,
                                const CUgraphEdgeData **data, size_t *ndeps) {
    (void)stream; (void)deps; (void)data; (void)ndeps;
    if (capture_info_result != CUDA_SUCCESS) return capture_info_result;
    *status = capturing != NULL ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
    *graph = capturing; *id = capturing != NULL ? capture_id : 0;
    return CUDA_SUCCESS;
}
static CUresult driver_alloc_async(CUdeviceptr *ptr, size_t bytes, CUstream stream) {
    (void)bytes; (void)stream; nodes++; *ptr = ++serial * MIB; return add_result;
}
static CUresult driver_alloc_from_pool(CUdeviceptr *ptr, size_t bytes, CUmemoryPool pool, CUstream stream) {
    (void)pool;
    return driver_alloc_async(ptr, bytes, stream);
}
static void capture_pause(int *blocked) {
    pthread_mutex_lock(&submit_gate);
    capture_entered = 1;
    pthread_cond_broadcast(&submit_cond);
    while (*blocked) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
}
static CUresult driver_begin_capture(CUstream stream, CUstreamCaptureMode mode) {
    (void)stream;
    if (capture_begin_result != CUDA_SUCCESS) return capture_begin_result;
    capturing = (CUgraph)++serial;
    capture_id = (cuuint64_t)(uintptr_t)capturing;
    capturing_mode = mode;
    if (mode == CU_STREAM_CAPTURE_MODE_GLOBAL) atomic_fetch_add(&global_captures, 1);
    capture_pause(&block_capture_begin);
    return CUDA_SUCCESS;
}
static CUresult driver_end_capture(CUstream stream, CUgraph *graph) {
    (void)stream;
    *graph = capture_end_result == CUDA_SUCCESS ? capturing : NULL;
    if (!keep_capture) {
        if (capturing_mode == CU_STREAM_CAPTURE_MODE_GLOBAL) atomic_fetch_sub(&global_captures, 1);
        capturing = NULL;
        capture_pause(&block_capture_end);
    }
    return capture_end_result;
}

static CUgraph make_graph(void) { return (CUgraph)++serial; }
static CUresult add(CUgraph graph, size_t mib, int with_free, CUdeviceptr *ptr) {
    CUDA_MEM_ALLOC_NODE_PARAMS params = {0};
    params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    params.bytesize = mib * MIB;
    CUgraphNode node, free_node;
    CUresult result = cuGraphAddMemAllocNode(&node, graph, NULL, 0, &params);
    if (result != CUDA_SUCCESS) return result;
    if (ptr != NULL) *ptr = params.dptr;
    return with_free ? cuGraphAddMemFreeNode(&free_node, graph, &node, 1, params.dptr) : CUDA_SUCCESS;
}
static CUgraphExec instantiate(CUgraph graph) {
    CUgraphExec exec;
    REQUIRE(cuGraphInstantiateWithFlags(&exec, graph, 0) == CUDA_SUCCESS);
    return exec;
}
static void destroy(CUgraph graph, CUgraphExec exec) {
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    REQUIRE(cuGraphExecDestroy(exec) == CUDA_SUCCESS);
}
static void clean(void) {
    gpu_done = 1;
    REQUIRE(oom_check(0, 0) == 0);
    REQUIRE(used == 256 * MIB);
    gpu_done = 0;
}

static void test_node_failure(void) {
    CUgraph graph = make_graph();
    int before = nodes;
    REQUIRE(add(graph, 400, 0, NULL) == CUDA_SUCCESS);
    REQUIRE(add(graph, 400, 0, NULL) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(nodes == before + 1); /* Refused nodes never reach the driver. */
    REQUIRE(used == 656 * MIB);
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    clean();
    graph = make_graph();
    add_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(add(graph, 600, 0, NULL) == add_result);
    REQUIRE(used == 256 * MIB); /* Native failure refunds the reservation. */
    add_result = CUDA_SUCCESS;
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
}

static void test_serial_and_overlap(void) {
    CUgraph a = make_graph(), b = make_graph();
    REQUIRE(add(a, 300, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(add(b, 300, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec x = instantiate(a), y = instantiate(b);
    REQUIRE(cuGraphLaunch(x, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(cuGraphLaunch(y, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(used == 556 * MIB); /* Same stream, neither event complete. */
    REQUIRE(reserve_device_memory(0, 300 * MIB) == 0);
    release_device_memory(0, 300 * MIB);
    gpu_done = 1;
    REQUIRE(oom_check(0, 0) == 0);
    gpu_done = 0;
    REQUIRE(cuGraphLaunch(x, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(cuGraphLaunch(y, (CUstream)20) == CUDA_SUCCESS);
    REQUIRE(used == 856 * MIB); /* Separate streams need both allocations. */
    REQUIRE(cuGraphLaunch(x, (CUstream)20) == CUDA_SUCCESS);
    REQUIRE(used == 856 * MIB); /* x's previous launch can still overlap y. */
    pool_result = CUDA_ERROR_INVALID_VALUE;
    gpu_done = 1;
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(used == 856 * MIB); /* A failed pool query cannot justify a refund. */
    pool_result = CUDA_SUCCESS;
    reserved = 640 * MIB;
    gpu_done = 1;
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(used == 896 * MIB); /* Completion does not free cached backing. */
    destroy(a, x); destroy(b, y);
    clean();
}

static void test_capture_admission(int from_pool) {
    CUgraph running = make_graph();
    REQUIRE(add(running, 64, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(running);
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    capturing = make_graph();
    int before = nodes;
    CUdeviceptr p;
    CUresult result = from_pool
        ? cuMemAllocFromPoolAsync(&p, 400 * MIB, (CUmemoryPool)1, (CUstream)20)
        : cuMemAllocAsync(&p, 400 * MIB, (CUstream)20);
    REQUIRE(result == CUDA_SUCCESS);
    graph_ptr_freed_async(p, (CUstream)20);
    result = from_pool
        ? cuMemAllocFromPoolAsync(&p, 400 * MIB, (CUmemoryPool)1, (CUstream)20)
        : cuMemAllocAsync(&p, 400 * MIB, (CUstream)20);
    REQUIRE(result == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(nodes == before + 1);
    add_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(cuMemAllocAsync(&p, 16 * MIB, (CUstream)20) == add_result);
    add_result = CUDA_SUCCESS;
    REQUIRE(used == 656 * MIB);
    CUgraph captured = capturing;
    capturing = NULL;
    graph_capture_end(123, captured);
    REQUIRE(cuGraphDestroy(captured) == CUDA_SUCCESS);
    destroy(running, exec);
    clean();
}

static void test_event_failure(void) {
    CUgraph graph = make_graph();
    REQUIRE(add(graph, 600, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec;
    int before = submissions;
    exec = instantiate(graph); /* Instantiation needs no completion event. */
    event_result = CUDA_ERROR_INVALID_CONTEXT;
    REQUIRE(cuGraphLaunch(exec, NULL) == event_result);
    REQUIRE(submissions == before);
    event_result = CUDA_SUCCESS;
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    destroy(graph, exec);
    clean();
}

static void test_per_thread_capture(int default_stream, int failed) {
    CUstream stream = default_stream ? NULL : (CUstream)10;
    REQUIRE(cuStreamBeginCapture_v2_ptsz(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    CUdeviceptr ptr;
    REQUIRE(cuMemAllocAsync(&ptr, 400 * MIB, default_stream ? CU_STREAM_PER_THREAD : stream) == CUDA_SUCCESS);
    REQUIRE(used == 656 * MIB);
    CUgraph graph = NULL;
    capture_end_result = failed ? CUDA_ERROR_STREAM_CAPTURE_INVALIDATED : CUDA_SUCCESS;
    REQUIRE(cuStreamEndCapture_ptsz(stream, &graph) == capture_end_result);
    if (!failed) REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    REQUIRE(used == 256 * MIB);
    REQUIRE(reserve_device_memory(0, 600 * MIB) == 0);
    release_device_memory(0, 600 * MIB);
    capture_end_result = CUDA_SUCCESS;
    clean();
}

static void *capture_thread(void *mode) {
    REQUIRE(cuStreamBeginCapture_v2_ptsz(NULL, (CUstreamCaptureMode)(uintptr_t)mode) == CUDA_SUCCESS);
    CUdeviceptr ptr;
    REQUIRE(cuMemAllocAsync(&ptr, 200 * MIB, CU_STREAM_PER_THREAD) == CUDA_SUCCESS);
    pthread_mutex_lock(&submit_gate);
    capture_ready = 1;
    pthread_cond_broadcast(&submit_cond);
    while (!capture_finish) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
    CUgraph graph;
    REQUIRE(cuStreamEndCapture_ptsz(NULL, &graph) == capture_end_result);
    return NULL;
}

static void test_per_thread_capture_ids(CUstreamCaptureMode mode) {
    CUgraph running = make_graph();
    REQUIRE(add(running, 64, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(running);
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(cuStreamBeginCapture_v2_ptsz(NULL, mode) == CUDA_SUCCESS);
    CUdeviceptr ptr;
    REQUIRE(cuMemAllocAsync(&ptr, 100 * MIB, CU_STREAM_PER_THREAD) == CUDA_SUCCESS);
    capture_end_result = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    capture_ready = 0; capture_finish = 0;
    pthread_t thread;
    REQUIRE(pthread_create(&thread, NULL, capture_thread, (void *)(uintptr_t)mode) == 0);
    pthread_mutex_lock(&submit_gate);
    while (!capture_ready) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
    CUgraph graph;
    REQUIRE(cuStreamEndCapture_ptsz(NULL, &graph) == capture_end_result);
    REQUIRE(used == 456 * MIB);
    pthread_mutex_lock(&submit_gate);
    capture_finish = 1;
    pthread_cond_broadcast(&submit_cond);
    pthread_mutex_unlock(&submit_gate);
    REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(used == 320 * MIB);
    capture_end_result = CUDA_SUCCESS;
    destroy(running, exec);
    clean();
}

static void *capture_window(void *mode) {
    CUgraph graph;
    REQUIRE(cuStreamBeginCapture_v2((CUstream)20, (CUstreamCaptureMode)(uintptr_t)mode) == CUDA_SUCCESS);
    REQUIRE(cuStreamEndCapture((CUstream)20, &graph) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    return NULL;
}

static void test_capture_window(CUstreamCaptureMode mode, int ending) {
    CUgraph a = make_graph(), b = make_graph(), mutation = make_graph();
    REQUIRE(add(a, 64, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(add(b, 64, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec x = instantiate(a), y = instantiate(b);
    REQUIRE(cuGraphLaunch(x, (CUstream)10) == CUDA_SUCCESS);
    capture_entered = 0;
    block_capture_begin = !ending; block_capture_end = ending;
    pthread_t thread;
    REQUIRE(pthread_create(&thread, NULL, capture_window, (void *)(uintptr_t)mode) == 0);
    pthread_mutex_lock(&submit_gate);
    while (!capture_entered) pthread_cond_wait(&submit_cond, &submit_gate);
    if (ending) {
        while (atomic_load(&global_captures) != 0) pthread_cond_wait(&submit_cond, &submit_gate);
    }
    pthread_mutex_unlock(&submit_gate);
    int before = event_queries;
    REQUIRE(cuGraphLaunch(y, (CUstream)30) == CUDA_SUCCESS);
    REQUIRE(add(mutation, 16, 0, NULL) == CUDA_SUCCESS);
    REQUIRE(cuDeviceGraphMemTrim(0) == CUDA_SUCCESS);
    REQUIRE(oom_check(0, 0) == 0);
    if (mode == CU_STREAM_CAPTURE_MODE_GLOBAL) REQUIRE(event_queries == before);
    else REQUIRE(event_queries > before);
    CUgraph reused;
    if (ending) REQUIRE(cuStreamBeginCapture_v2((CUstream)20, mode) == CUDA_SUCCESS);
    pthread_mutex_lock(&submit_gate);
    block_capture_begin = 0; block_capture_end = 0;
    pthread_cond_broadcast(&submit_cond);
    pthread_mutex_unlock(&submit_gate);
    REQUIRE(pthread_join(thread, NULL) == 0);
    if (ending) {
        REQUIRE(cuStreamEndCapture((CUstream)20, &reused) == CUDA_SUCCESS);
        REQUIRE(cuGraphDestroy(reused) == CUDA_SUCCESS);
    }
    before = event_queries;
    REQUIRE(oom_check(0, 0) == 0);
    REQUIRE(event_queries > before);
    REQUIRE(cuGraphDestroy(mutation) == CUDA_SUCCESS);
    destroy(a, x); destroy(b, y); clean();
}

static void *capture_poll(void *unused) {
    (void)unused;
    REQUIRE(oom_check(0, 0) == 0);
    return NULL;
}

static void test_capture_failure(void) {
    CUgraph running = make_graph(), graph;
    REQUIRE(add(running, 64, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(running);
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    capture_begin_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(cuStreamBeginCapture_v2((CUstream)20, CU_STREAM_CAPTURE_MODE_GLOBAL) == capture_begin_result);
    capture_begin_result = CUDA_SUCCESS;
    int before = event_queries;
    REQUIRE(oom_check(0, 0) == 0);
    REQUIRE(event_queries > before);
    REQUIRE(cuStreamBeginCapture_v2((CUstream)20, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    capture_end_result = CUDA_ERROR_INVALID_VALUE; keep_capture = 1;
    capture_info_result = CUDA_ERROR_UNKNOWN;
    REQUIRE(cuStreamEndCapture((CUstream)20, &graph) == capture_end_result);
    pthread_t thread;
    before = event_queries;
    REQUIRE(pthread_create(&thread, NULL, capture_poll, NULL) == 0);
    REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(event_queries == before);
    capture_info_result = CUDA_SUCCESS;
    capture_end_result = CUDA_SUCCESS; keep_capture = 0;
    REQUIRE(cuStreamEndCapture((CUstream)20, &graph) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    destroy(running, exec); clean();
}

static void test_invalidated_capture_info_failure(void) {
    CUgraph running = make_graph(), graph;
    REQUIRE(add(running, 64, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(running);
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(cuStreamBeginCapture_v2((CUstream)20, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    CUdeviceptr ptr;
    REQUIRE(cuMemAllocAsync(&ptr, 400 * MIB, (CUstream)20) == CUDA_SUCCESS);
    capture_end_result = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    capture_info_result = CUDA_ERROR_UNKNOWN;
    int before = event_queries;
    REQUIRE(cuStreamEndCapture((CUstream)20, &graph) == capture_end_result);
    REQUIRE(used == 320 * MIB && event_queries > before);
    capture_info_result = CUDA_SUCCESS; capture_end_result = CUDA_SUCCESS;
    REQUIRE(reserve_device_memory(0, 600 * MIB) == CUDA_SUCCESS);
    release_device_memory(0, 600 * MIB);
    destroy(running, exec); clean();
}

static void test_context_lifetime(void) {
    CUgraphExec exec;
    /* Instantiate in context 2, submit on context 1's stream. */
    CUgraph graph = make_graph();
    REQUIRE(add(graph, 300, 1, NULL) == CUDA_SUCCESS);
    current_ctx = (CUcontext)2;
    exec = instantiate(graph);
    current_ctx = (CUcontext)1;
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    /* Switch to another context while keeping the caller's context intact. */
    REQUIRE(cuGraphLaunch(exec, (CUstream)30) == CUDA_SUCCESS);
    REQUIRE(current_ctx == (CUcontext)1);
    gpu_done = 1;
    destroy(graph, exec);
    clean();

    graph = make_graph();
    REQUIRE(add(graph, 300, 1, NULL) == CUDA_SUCCESS);
    exec = instantiate(graph);
    REQUIRE(cuGraphLaunch(exec, (CUstream)30) == CUDA_SUCCESS);
    destroy(graph, exec); /* Retired, with GPU work still in flight. */
    REQUIRE(used == 556 * MIB);
    for (int i = 0; i < 8192; i++) {
        if (events[i].ctx == (CUcontext)2) events[i].alive = 0;
    }
    graph_context_destroyed((CUcontext)2, 2);
    REQUIRE(used == 256 * MIB); /* Never query an already destroyed event again. */
    clean();
}

static void test_reused_context_handle(void) {
    CUgraph old = make_graph(); CUgraphExec old_exec;
    REQUIRE(add(old, 300, 1, NULL) == CUDA_SUCCESS); old_exec = instantiate(old);
    REQUIRE(cuGraphLaunch(old_exec, (CUstream)30) == CUDA_SUCCESS);
    destroy(old, old_exec);
    /* Native destruction finished, but its hook has not yet notified us. */
    for (int i = 0; i < 8192; i++) {
        if (events[i].ctx_id == 2) events[i].alive = 0;
    }
    context_ids[2] = 22; /* The same raw context handle now names a new context. */
    CUgraph graph = make_graph(), other = make_graph();
    REQUIRE(add(graph, 400, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(add(other, 400, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph), next = instantiate(other);
    REQUIRE(cuGraphLaunch(exec, (CUstream)30) == CUDA_SUCCESS);
    graph_context_destroyed((CUcontext)2, 2); /* Late cleanup of the OLD context. */
    REQUIRE(used == 656 * MIB);
    REQUIRE(cuGraphLaunch(next, (CUstream)20) == CUDA_ERROR_OUT_OF_MEMORY);
    gpu_done = 1; destroy(graph, exec); destroy(other, next); clean();
    context_ids[2] = 2;
}

static void test_record_failure_sync(void) {
    CUgraph graph = make_graph();
    REQUIRE(add(graph, 600, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph);
    record_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    record_result = CUDA_SUCCESS;
    destroy(graph, exec);
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_ERROR_OUT_OF_MEMORY);
    gpu_done = 1;
    graph_synchronized(NULL, (CUstream)10, 0, graph_sync_begin());
    clean();
}

static void test_stream_fallback(void) {
    void *native_id = cuda_library_entry[CUDA_OVERRIDE_ENUM(cuStreamGetId)].fn_ptr;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuStreamGetId)].fn_ptr = NULL;
    CUgraph graphs[3]; CUgraphExec execs[3];
    for (int i = 0; i < 3; i++) {
        graphs[i] = make_graph();
        REQUIRE(add(graphs[i], 400, 1, NULL) == CUDA_SUCCESS);
        execs[i] = instantiate(graphs[i]);
    }
    REQUIRE(cuGraphLaunch(execs[0], (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(cuGraphLaunch(execs[1], (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(used == 656 * MIB);
    graph_stream_destroyed((CUstream)10);
    /* A new stream reusing the same raw handle must not alias queued work. */
    REQUIRE(cuGraphLaunch(execs[2], (CUstream)10) == CUDA_ERROR_OUT_OF_MEMORY);
    gpu_done = 1;
    REQUIRE(cuGraphLaunch(execs[2], (CUstream)10) == CUDA_SUCCESS);
    for (int i = 0; i < 3; i++) destroy(graphs[i], execs[i]);
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuStreamGetId)].fn_ptr = native_id;
    clean();
}

static void test_cached_replay(void) {
    CUgraph graphs[128]; CUgraphExec execs[128];
    int before = events_created;
    for (int i = 0; i < 128; i++) {
        graphs[i] = make_graph();
        REQUIRE(add(graphs[i], 1, 1, NULL) == CUDA_SUCCESS);
        execs[i] = instantiate(graphs[i]);
    }
    REQUIRE(events_created == before);
    REQUIRE(cuGraphLaunch(execs[0], (CUstream)10) == CUDA_SUCCESS);
    int queries = event_queries, pools = pool_queries;
    for (int i = 0; i < 1000; i++) REQUIRE(cuGraphLaunch(execs[0], (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(event_queries == queries && pool_queries == pools);
    REQUIRE(events_created == before + 1);
    gpu_done = 1;
    for (int i = 0; i < 128; i++) destroy(graphs[i], execs[i]);
    clean();
}

static void *replay_thread(void *exec) {
    for (int i = 0; i < 50; i++) REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    return NULL;
}
static void *submit_once(void *exec) {
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    return NULL;
}
static void *sync_once(void *whole_context) {
    if (whole_context != NULL) REQUIRE(cuCtxSynchronize() == CUDA_SUCCESS);
    else REQUIRE(cuStreamSynchronize((CUstream)10) == CUDA_SUCCESS);
    return NULL;
}

static void test_sync_submission_race(int whole_context, int already_pending) {
    CUgraph a = make_graph(), b = make_graph();
    REQUIRE(add(a, 400, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(add(b, 400, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec x = instantiate(a), y = instantiate(b);
    pthread_t syncing, submitting;
    block_sync = 1; sync_entered = 0;
    if (already_pending) {
        block_submit = 1; submit_entered = 0;
        REQUIRE(pthread_create(&submitting, NULL, submit_once, x) == 0);
        pthread_mutex_lock(&submit_gate);
        while (!submit_entered) pthread_cond_wait(&submit_cond, &submit_gate);
        pthread_mutex_unlock(&submit_gate);
    }
    REQUIRE(pthread_create(&syncing, NULL, sync_once,
                         whole_context ? (void *)1 : NULL) == 0);
    pthread_mutex_lock(&submit_gate);
    while (!sync_entered) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
    /* Preserve new submissions and calls pending at sync entry until GPU completion. */
    if (already_pending) {
        pthread_mutex_lock(&submit_gate);
        block_submit = 0;
        pthread_cond_broadcast(&submit_cond);
        pthread_mutex_unlock(&submit_gate);
        REQUIRE(pthread_join(submitting, NULL) == 0);
    } else {
        REQUIRE(cuGraphLaunch(x, (CUstream)10) == CUDA_SUCCESS);
    }
    pthread_mutex_lock(&submit_gate);
    block_sync = 0;
    pthread_cond_broadcast(&submit_cond);
    pthread_mutex_unlock(&submit_gate);
    REQUIRE(pthread_join(syncing, NULL) == 0);
    REQUIRE(cuGraphLaunch(y, (CUstream)20) == CUDA_ERROR_OUT_OF_MEMORY);
    gpu_done = 1;
    destroy(a, x); destroy(b, y); clean();
}
static void test_host_submission_lifetime(void) {
    CUgraph graph = make_graph();
    REQUIRE(add(graph, 600, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph);
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) REQUIRE(pthread_create(&threads[i], NULL, replay_thread, exec) == 0);
    for (int i = 0; i < 4; i++) REQUIRE(pthread_join(threads[i], NULL) == 0);
    gpu_done = 1;
    destroy(graph, exec);
    clean();

    graph = make_graph();
    REQUIRE(add(graph, 600, 1, NULL) == CUDA_SUCCESS);
    exec = instantiate(graph);
    block_submit = 1; submit_entered = 0;
    REQUIRE(pthread_create(&threads[0], NULL, submit_once, exec) == 0);
    pthread_mutex_lock(&submit_gate);
    while (!submit_entered) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
    /* Native destroy can finish before our launch wrapper has returned. */
    destroy(graph, exec);
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_ERROR_OUT_OF_MEMORY);
    pthread_mutex_lock(&submit_gate);
    block_submit = 0;
    pthread_cond_broadcast(&submit_cond);
    pthread_mutex_unlock(&submit_gate);
    REQUIRE(pthread_join(threads[0], NULL) == 0);
    clean();
}

static void test_persistent(int mode) {
    CUgraph a = make_graph(), b = make_graph();
    CUdeviceptr p, q;
    REQUIRE(add(a, 512, 0, &p) == CUDA_SUCCESS);
    REQUIRE(add(b, 512, 0, &q) == CUDA_SUCCESS);
    CUgraphExec x, y;
    if (mode == 2) {
        CUDA_GRAPH_INSTANTIATE_PARAMS params = {0};
        params.flags = CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD;
        REQUIRE(cuGraphInstantiateWithParams(&x, a, &params) == CUDA_SUCCESS);
        int before = instantiations;
        REQUIRE(cuGraphInstantiateWithParams(&y, b, &params) == CUDA_ERROR_OUT_OF_MEMORY);
        REQUIRE(instantiations == before);
        REQUIRE(cuGraphDestroy(b) == CUDA_SUCCESS);
    } else {
        x = instantiate(a); y = instantiate(b);
        CUresult (*submit)(CUgraphExec, CUstream) = mode == 1 ? cuGraphUpload : cuGraphLaunch;
        REQUIRE(submit(x, NULL) == CUDA_SUCCESS);
        int before = submissions;
        REQUIRE(submit(y, NULL) == CUDA_ERROR_OUT_OF_MEMORY);
        REQUIRE(submissions == before); /* Reject before submitting GPU work. */
        destroy(b, y);
    }
    REQUIRE(used == 768 * MIB);
    destroy(a, x);
    graph_ptr_freed(p); graph_ptr_freed(q);
    clean();
}

static void test_submission_failure(void) {
    CUgraph a = make_graph(), b = make_graph();
    CUdeviceptr p, q;
    REQUIRE(add(a, 400, 0, &p) == CUDA_SUCCESS);
    REQUIRE(add(b, 300, 0, &q) == CUDA_SUCCESS);
    CUgraphExec x = instantiate(a), y = instantiate(b);
    REQUIRE(cuGraphLaunch(x, NULL) == CUDA_SUCCESS);
    submit_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(cuGraphLaunch(y, NULL) == submit_result);
    submit_result = CUDA_SUCCESS;
    REQUIRE(used == 656 * MIB);
    destroy(a, x); destroy(b, y);
    graph_ptr_freed(p); graph_ptr_freed(q);
    clean();
}

static void test_persistent_replay_after_free(void) {
    CUgraph graph = make_graph(); CUdeviceptr ptr;
    REQUIRE(add(graph, 512, 0, &ptr) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph);
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(used == 768 * MIB);
    reserved = 512 * MIB;
    graph_ptr_freed(ptr); /* Preserve metadata, release the live allocation. */
    REQUIRE(used == 768 * MIB); /* The driver still retains the physical pool. */
    gpu_done = 1;
    REQUIRE(cuDeviceGraphMemTrim(0) == CUDA_SUCCESS);
    REQUIRE(used == 256 * MIB);
    REQUIRE(reserve_device_memory(0, 600 * MIB) == 0);
    int before = submissions;
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(submissions == before);
    release_device_memory(0, 600 * MIB);
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(used == 768 * MIB);
    destroy(graph, exec); graph_ptr_freed(ptr);
    clean();
}

static void test_destroy_inflight(void) {
    CUgraph graph = make_graph();
    REQUIRE(add(graph, 600, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph);
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    reserved = 640 * MIB;
    destroy(graph, exec);
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_ERROR_OUT_OF_MEMORY);
    clean(); /* Completion and trim release even the retired exec's charge. */
}

int main(void) {
#define DRIVER(name, fn) cuda_library_entry[CUDA_OVERRIDE_ENUM(name)].fn_ptr = (void *)(fn)
    DRIVER(cuGraphAddMemAllocNode, driver_add);
    DRIVER(cuGraphAddMemFreeNode, driver_free_node);
    DRIVER(cuGraphInstantiateWithFlags, driver_instantiate);
    DRIVER(cuGraphInstantiateWithParams, driver_params);
    DRIVER(cuGraphLaunch, driver_submit);
    DRIVER(cuGraphUpload, driver_submit);
    DRIVER(cuGraphDestroy, driver_destroy);
    DRIVER(cuGraphExecDestroy, driver_destroy);
    DRIVER(cuEventCreate, driver_event_create);
    DRIVER(cuEventRecord, driver_event_record);
    DRIVER(cuEventQuery, driver_event_query);
    DRIVER(cuEventDestroy_v2, driver_event_destroy);
    DRIVER(cuDeviceGetGraphMemAttribute, driver_reserved);
    DRIVER(cuDeviceGraphMemTrim, driver_trim);
    DRIVER(cuStreamGetId, driver_stream_id);
    DRIVER(cuStreamGetCtx, driver_stream_ctx);
    DRIVER(cuCtxGetCurrent, driver_current);
    DRIVER(cuCtxGetDevice, driver_device);
    DRIVER(cuCtxSynchronize, driver_context_sync);
    DRIVER(cuStreamSynchronize, driver_stream_sync);
    DRIVER(cuCtxGetId, driver_context_id);
    DRIVER(cuCtxPushCurrent_v2, driver_push);
    DRIVER(cuCtxPopCurrent_v2, driver_pop);
    DRIVER(cuStreamGetCaptureInfo_v3, driver_capture);
    DRIVER(cuMemAllocAsync, driver_alloc_async);
    DRIVER(cuMemAllocFromPoolAsync, driver_alloc_from_pool);
    DRIVER(cuStreamBeginCapture_v2_ptsz, driver_begin_capture);
    DRIVER(cuStreamEndCapture_ptsz, driver_end_capture);
    DRIVER(cuStreamBeginCapture_v2, driver_begin_capture);
    DRIVER(cuStreamEndCapture, driver_end_capture);
    test_node_failure();
    for (int from_pool = 0; from_pool < 2; from_pool++) test_capture_admission(from_pool);
    for (int default_stream = 0; default_stream < 2; default_stream++)
        for (int failed = 0; failed < 2; failed++) test_per_thread_capture(default_stream, failed);
    test_per_thread_capture_ids(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    test_per_thread_capture_ids(CU_STREAM_CAPTURE_MODE_GLOBAL);
    test_capture_window(CU_STREAM_CAPTURE_MODE_GLOBAL, 0);
    test_capture_window(CU_STREAM_CAPTURE_MODE_GLOBAL, 1);
    test_capture_window(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, 0);
    test_capture_failure();
    test_invalidated_capture_info_failure();
    test_event_failure();
    test_serial_and_overlap();
    for (int mode = 0; mode < 3; mode++) test_persistent(mode);
    test_submission_failure();
    test_persistent_replay_after_free();
    test_destroy_inflight();
    test_context_lifetime();
    test_reused_context_handle();
    test_record_failure_sync();
    test_stream_fallback();
    test_cached_replay();
    test_host_submission_lifetime();
    for (int context = 0; context < 2; context++)
        for (int pending = 0; pending < 2; pending++)
            test_sync_submission_race(context, pending);
    puts("PASS graph memory admission, overlap, retained pool and failure rollback");
    return 0;
}

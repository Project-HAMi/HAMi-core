#pragma once
// Copyright 2026 The HAMi Authors.
/* Exercise production accounting with an asynchronous fake driver. */
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include "allocator/allocator.h"
#include "include/hostpid_broker.h"
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
static int nodes, instantiations, gpu_done;
static _Atomic int submissions;
static CUresult add_result, submit_result, event_result, pool_result, record_result;
static _Thread_local CUgraph capturing;
static _Thread_local cuuint64_t capture_id = 123;
static _Thread_local CUstreamCaptureMode capturing_mode;
static _Atomic int global_captures;
static CUcontext current_ctx = (CUcontext)1, pushed_ctx;
static struct { CUcontext ctx; cuuint64_t ctx_id; CUstream stream; int alive; } events[8192];
static CUstream completed_stream;
static struct {
    CUgraph graph, child;
    CUgraphNodeType type;
    CUdeviceptr ptr;
    unsigned int grid, block;
} graph_nodes[8192];
static CUresult graph_query_result;
static int graph_node_queries;
static cuuint64_t context_ids[] = {0, 1, 2};
static int event_queries, pool_queries, events_created, trim_calls;
static _Thread_local CUstreamCaptureMode interaction_mode;
static _Thread_local int external_capture;
static CUresult exchange_result;
/** Model native capture interaction mode without changing the capture itself. */
static CUresult driver_exchange_mode(CUstreamCaptureMode *mode) {
    if (exchange_result != CUDA_SUCCESS) return exchange_result;
    CUstreamCaptureMode old = interaction_mode; interaction_mode = *mode; *mode = old;
    return CUDA_SUCCESS;
}

static pthread_mutex_t submit_gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t submit_cond = PTHREAD_COND_INITIALIZER;
static _Atomic int block_submit;
static int submit_entered;
static int block_sync, sync_entered;
static CUresult capture_end_result;
static CUresult capture_begin_result;
static CUresult capture_info_result;
static int keep_capture, block_capture_begin, block_capture_end, capture_entered;
static int capture_ready, capture_finish;

/* Graph launches go through the compute limiter; it is off unless a test sets pidfound. */
int pidfound;
static int64_t limited_threads;
static int limited_launches;
void pre_launch_kernel(void) {}
int is_gpu_core_limit_enabled(void) { return 0; }
int wait_status_self(int status) { (void)status; return 1; }
void rate_limiter(int64_t grids, int64_t blocks) {
    limited_threads += grids * blocks;
    limited_launches++;
}
void ensure_initialized(void) {}
void ensure_hostpid(void) {}
/* context.c asks for driver admission; the broker is off in these tests. */
int driver_admission_enabled(void) { return 0; }
int driver_admission_acquire(hostpid_driver_admission_operation_t operation) {
    (void)operation;
    return -1;
}
void driver_admission_release(int *lease_fd) { *lease_fd = -1; }
void driver_admission_cleanup(void *lease_fd) { driver_admission_release(lease_fd); }
void lock_shrreg(void) {}
void unlock_shrreg(void) {}
unsigned int cuda_to_nvml_map(unsigned int dev) { return dev; }
uint64_t get_current_device_memory_limit(const int dev) { REQUIRE(dev == 0); return 1024 * MIB; }
size_t get_gpu_memory_usage(const int dev) { REQUIRE(dev == 0); return used; }
int clear_proc_slot_nolock(int force) { (void)force; return 0; }
int add_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid; (void)type; REQUIRE(dev == 0); used += bytes; return 0;
}
/** Subtract a simulated device charge after validating the device and available usage. */
int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t bytes, int type) {
    (void)pid; (void)type; REQUIRE(dev == 0 && used >= bytes); used -= bytes; return 0;
}
static CUresult driver_device_count(int *count) { *count = 1; return CUDA_SUCCESS; }
static CUresult driver_device(CUdevice *dev) { *dev = 0; return CUDA_SUCCESS; }

/** Simulate graph allocation node creation or an injected native failure. */
static CUresult driver_add(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                            size_t ndeps, CUDA_MEM_ALLOC_NODE_PARAMS *params) {
    (void)graph; (void)deps; (void)ndeps;
    if (add_result != CUDA_SUCCESS) return add_result;
    nodes++;
    *node = (CUgraphNode)++serial;
    params->dptr = ++serial * MIB;
    graph_nodes[(uintptr_t)*node].graph = graph;
    graph_nodes[(uintptr_t)*node].type = CU_GRAPH_NODE_TYPE_MEM_ALLOC;
    return CUDA_SUCCESS;
}
static CUresult driver_free_node(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                                  size_t ndeps, CUdeviceptr ptr) {
    (void)graph; (void)deps; (void)ndeps; (void)ptr;
    *node = (CUgraphNode)++serial;
    graph_nodes[serial].graph = graph;
    graph_nodes[serial].type = CU_GRAPH_NODE_TYPE_MEM_FREE;
    graph_nodes[serial].ptr = ptr;
    return CUDA_SUCCESS;
}

/** Return the nodes currently owned by a simulated graph. */
static CUresult driver_graph_nodes(CUgraph graph, CUgraphNode *nodes_out, size_t *count) {
    graph_node_queries++;
    if (graph_query_result != CUDA_SUCCESS) return graph_query_result;
    size_t found = 0;
    for (size_t i = 0; i < 8192; i++) {
        if (graph_nodes[i].graph != graph) continue;
        if (nodes_out != NULL && found < *count) nodes_out[found] = (CUgraphNode)i;
        found++;
    }
    *count = found;
    return CUDA_SUCCESS;
}
static CUresult driver_node_type(CUgraphNode node, CUgraphNodeType *type) {
    REQUIRE((uintptr_t)node < 8192); *type = graph_nodes[(uintptr_t)node].type; return CUDA_SUCCESS;
}
static CUresult driver_free_params(CUgraphNode node, CUdeviceptr *ptr) {
    REQUIRE((uintptr_t)node < 8192); *ptr = graph_nodes[(uintptr_t)node].ptr; return CUDA_SUCCESS;
}
static CUresult driver_child_graph(CUgraphNode node, CUgraph *graph) {
    REQUIRE((uintptr_t)node < 8192); *graph = graph_nodes[(uintptr_t)node].child; return CUDA_SUCCESS;
}
static CUresult driver_kernel_params(CUgraphNode node, CUDA_KERNEL_NODE_PARAMS *params) {
    REQUIRE((uintptr_t)node < 8192);
    *params = (CUDA_KERNEL_NODE_PARAMS){0};
    params->gridDimX = graph_nodes[(uintptr_t)node].grid; params->gridDimY = 1; params->gridDimZ = 1;
    params->blockDimX = graph_nodes[(uintptr_t)node].block; params->blockDimY = 1; params->blockDimZ = 1;
    return CUDA_SUCCESS;
}
/** Record generic child and free nodes in the simulated native graph. */
static CUresult driver_generic_node(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                                    const CUgraphEdgeData *edges, size_t count, CUgraphNodeParams *params) {
    (void)edges;
    if (add_result != CUDA_SUCCESS) return add_result;
    if (params->type == CU_GRAPH_NODE_TYPE_MEM_FREE)
        return driver_free_node(node, graph, deps, count, params->free.dptr);
    REQUIRE(params->type == CU_GRAPH_NODE_TYPE_GRAPH);
    *node = (CUgraphNode)++serial;
    graph_nodes[serial].graph = graph;
    graph_nodes[serial].type = CU_GRAPH_NODE_TYPE_GRAPH;
    graph_nodes[serial].child = params->graph.graph;
    return CUDA_SUCCESS;
}
static void setup_graph_queries(void) {
    cuda_library_entry[OVERRIDE_cuGraphGetNodes].fn_ptr = driver_graph_nodes;
    cuda_library_entry[OVERRIDE_cuGraphNodeGetType].fn_ptr = driver_node_type;
    cuda_library_entry[OVERRIDE_cuGraphMemFreeNodeGetParams].fn_ptr = driver_free_params;
    cuda_library_entry[OVERRIDE_cuGraphChildGraphNodeGetGraph].fn_ptr = driver_child_graph;
    cuda_library_entry[OVERRIDE_cuGraphKernelNodeGetParams_v2].fn_ptr = driver_kernel_params;
    cuda_library_entry[OVERRIDE_cuGraphAddNode_v2].fn_ptr = driver_generic_node;
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
/** Simulate a graph submission with optional blocking for concurrency tests. */
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
/** Block simulated context synchronization until the test releases it. */
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
/** Retire a simulated graph and every child whose ownership it holds. */
static CUresult driver_destroy(void *handle) {
    if (handle == NULL) return CUDA_SUCCESS;
    for (size_t i = 0; i < 8192; i++) {
        if (graph_nodes[i].graph != handle) continue;
        if (graph_nodes[i].type == CU_GRAPH_NODE_TYPE_GRAPH) driver_destroy(graph_nodes[i].child);
        graph_nodes[i].graph = NULL;
    }
    return CUDA_SUCCESS;
}
/** Create a simulated completion event bound to the current context generation. */
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
/** Resolve the simulated context for an explicit or default stream. */
static CUcontext stream_ctx(CUstream stream) {
    if (stream == NULL || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD) return current_ctx;
    return stream == (CUstream)30 ? (CUcontext)2 : (CUcontext)1;
}
/** Validate the event's context and return the configured recording result. */
static CUresult driver_event_record(CUevent event, CUstream stream) {
    REQUIRE((uintptr_t)event < 8192);
    if (!events[(uintptr_t)event].alive || events[(uintptr_t)event].ctx != stream_ctx(stream))
        return CUDA_ERROR_INVALID_HANDLE;
    events[(uintptr_t)event].stream = stream;
    return record_result;
}
/** Report simulated GPU completion while rejecting event queries during capture. */
static CUresult driver_event_query(CUevent event) {
    REQUIRE(capturing == NULL && atomic_load(&global_captures) == 0);
    REQUIRE(!external_capture || interaction_mode == CU_STREAM_CAPTURE_MODE_RELAXED);
    event_queries++;
    REQUIRE((uintptr_t)event < 8192);
    if (!events[(uintptr_t)event].alive) return CUDA_ERROR_INVALID_HANDLE;
    return gpu_done || (completed_stream != NULL && events[(uintptr_t)event].stream == completed_stream)
        ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
/** Retire a simulated completion event after validating its lifetime. */
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
/** Return simulated graph-pool backing or the configured native query failure. */
static CUresult driver_reserved(CUdevice dev, CUgraphMem_attribute attr, void *value) {
    REQUIRE(capturing == NULL && atomic_load(&global_captures) == 0);
    REQUIRE(!external_capture || interaction_mode == CU_STREAM_CAPTURE_MODE_RELAXED);
    pool_queries++;
    REQUIRE(dev == 0 && attr == CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT);
    if (pool_result != CUDA_SUCCESS) return pool_result;
    cuuint64_t *bytes = value; *bytes = reserved; return CUDA_SUCCESS;
}
/** Release simulated graph-pool backing after GPU completion. */
static CUresult driver_trim(CUdevice dev) {
    REQUIRE(capturing == NULL);
    REQUIRE(dev == 0);
    trim_calls++;
    if (gpu_done) reserved = 0;
    return CUDA_SUCCESS;
}
/** Return the simulated capture state or an injected capture-info failure. */
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
/** Pause native capture entry or exit at a test-controlled synchronization point. */
static void capture_pause(int *blocked) {
    pthread_mutex_lock(&submit_gate);
    capture_entered = 1;
    pthread_cond_broadcast(&submit_cond);
    while (*blocked) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
}
/** Begin a simulated capture and pause at the configured entry boundary. */
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
/** End or preserve a simulated capture according to the configured native result. */
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
/** Add a test graph allocation and optionally add its matching free node. */
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
    REQUIRE(cuDeviceGraphMemTrim(0) == CUDA_SUCCESS);
    REQUIRE(oom_check(0, 0) == 0);
    REQUIRE(used == 256 * MIB);
    gpu_done = 0;
}

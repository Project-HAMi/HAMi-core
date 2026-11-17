// Copyright 2026 The HAMi Authors.
#include <stdio.h>
#include <time.h>

#include "test/graph_accounting_fixture.h"


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
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_SUCCESS);
    REQUIRE(used == 856 * MIB); /* Quota pressure trims completed backing before retrying. */
    release_device_memory(0, 300 * MIB);
    destroy(a, x); destroy(b, y);
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









/** Verify context release and reset reclaim only the appropriate graph charges. */
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

/** Verify recycled context handles do not retain charges from an earlier generation. */
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

/** Verify missing stream IDs preserve conservative accounting across stream handle reuse. */
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
    REQUIRE(cuStreamDestroy_v2((CUstream)10) == CUDA_SUCCESS);
    /* A new stream reusing the same raw handle must not alias queued work. */
    REQUIRE(cuGraphLaunch(execs[2], (CUstream)10) == CUDA_ERROR_OUT_OF_MEMORY);
    gpu_done = 1;
    REQUIRE(cuGraphLaunch(execs[2], (CUstream)10) == CUDA_SUCCESS);
    for (int i = 0; i < 3; i++) destroy(graphs[i], execs[i]);
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuStreamGetId)].fn_ptr = native_id;
    clean();
}

/** Verify repeated graph replay uses cached accounting instead of scanning every live graph. */
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

/** Replay an executable graph repeatedly to exercise concurrent cached accounting. */
static void *replay_thread(void *exec) {
    for (int i = 0; i < 50; i++) REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    return NULL;
}
static void *submit_once(void *exec) {
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    return NULL;
}
/** Perform one stream or context synchronization in a test thread. */
static void *sync_once(void *whole_context) {
    if (whole_context != NULL) REQUIRE(cuCtxSynchronize() == CUDA_SUCCESS);
    else REQUIRE(cuStreamSynchronize((CUstream)10) == CUDA_SUCCESS);
    return NULL;
}

/** Verify synchronization preserves newer and still-pending graph submissions. */
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
/** Verify executable destruction and context cleanup respect in-flight host submissions. */
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

/** Verify persistent graph allocations are charged before launch, upload, or instantiation upload. */
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

#if CUDA_VERSION >= 12040
static CUresult driver_green_context(CUcontext *ctx, CUgreenCtx green) {
    (void)green; *ctx = (CUcontext)2; return CUDA_SUCCESS;
}

/** Invalidate the simulated event handles owned by the destroyed green context. */
static CUresult driver_green_destroy(CUgreenCtx green) {
    (void)green;
    for (int i = 0; i < 8192; i++) {
        if (events[i].ctx == (CUcontext)2) events[i].alive = 0;
    }
    return CUDA_SUCCESS;
}

/** Release in-flight tracking when a green context owns the completion event. */
static void test_green_context_lifetime(void) {
    CUgraph graph = make_graph();
    REQUIRE(add(graph, 300, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph);
    REQUIRE(cuGraphLaunch(exec, (CUstream)30) == CUDA_SUCCESS);
    destroy(graph, exec);
    REQUIRE(used == 556 * MIB);
    REQUIRE(cuGreenCtxDestroy((CUgreenCtx)2) == CUDA_SUCCESS);
    clean();
}
#endif

/** Return a failed stream lookup with an unusable output context. */
static CUresult failed_capture_context(CUstream stream, CUcontext *ctx) {
    (void)stream; *ctx = (CUcontext)99; return CUDA_ERROR_INVALID_HANDLE;
}

/** Ignore context outputs from failed lookups while retaining capture termination tracking. */
static void test_capture_context_failure(void) {
    cuda_library_entry[OVERRIDE_cuStreamGetCtx].fn_ptr = failed_capture_context;
    CUgraph graph;
    REQUIRE(cuStreamBeginCapture_v2((CUstream)10, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    REQUIRE(cuStreamEndCapture((CUstream)10, &graph) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    cuda_library_entry[OVERRIDE_cuStreamGetCtx].fn_ptr = driver_stream_ctx;
    clean();
}

/** Relax internal queries for captures entered through native APIs and restore the caller's mode. */
static void test_external_capture_queries(void) {
    CUgraph graph = make_graph();
    REQUIRE(add(graph, 300, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph);
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    for (int mode = 0; mode < 3; mode++) {
        interaction_mode = mode; external_capture = 1;
        int before = event_queries;
        REQUIRE(oom_check(0, 0) == 0);
        REQUIRE(event_queries > before && interaction_mode == mode);
        exchange_result = CUDA_ERROR_UNKNOWN;
        before = event_queries;
        REQUIRE(oom_check(0, 0) == 0 && event_queries == before);
        REQUIRE(interaction_mode == mode);
        exchange_result = CUDA_SUCCESS; external_capture = 0;
    }
    interaction_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
    destroy(graph, exec); clean();
}

static int block_destroy, destroy_entered, reused_ready, reused_done;
/** Pause native destruction after the handle becomes reusable. */
static CUresult driver_stream_destroy(CUstream stream) {
    (void)stream;
    pthread_mutex_lock(&submit_gate);
    destroy_entered = 1;
    pthread_cond_broadcast(&submit_cond);
    while (block_destroy) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
    return CUDA_SUCCESS;
}
static void *destroy_stream_once(void *unused) {
    (void)unused;
    REQUIRE(cuStreamDestroy_v2((CUstream)10) == CUDA_SUCCESS);
    return NULL;
}
static void *submit_reused_stream(void *exec) {
    pthread_mutex_lock(&submit_gate);
    reused_ready = 1; pthread_cond_broadcast(&submit_cond);
    pthread_mutex_unlock(&submit_gate);
    REQUIRE(cuGraphLaunch(exec, (CUstream)10) == CUDA_SUCCESS);
    pthread_mutex_lock(&submit_gate);
    reused_done = 1; pthread_cond_broadcast(&submit_cond);
    pthread_mutex_unlock(&submit_gate);
    return NULL;
}

/** Preserve ordering for a stream handle reused while old destruction is returning. */
static void test_stream_destroy_race(void) {
    void *native_id = cuda_library_entry[OVERRIDE_cuStreamGetId].fn_ptr;
    cuda_library_entry[OVERRIDE_cuStreamGetId].fn_ptr = NULL;
    CUgraph a = make_graph(), b = make_graph();
    REQUIRE(add(a, 400, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(add(b, 400, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec x = instantiate(a), y = instantiate(b);
    pthread_t destroying, submitting;
    block_destroy = 1; destroy_entered = reused_ready = reused_done = 0;
    REQUIRE(pthread_create(&destroying, NULL, destroy_stream_once, NULL) == 0);
    pthread_mutex_lock(&submit_gate);
    while (!destroy_entered) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
    REQUIRE(pthread_create(&submitting, NULL, submit_reused_stream, x) == 0);
    pthread_mutex_lock(&submit_gate);
    while (!reused_ready) pthread_cond_wait(&submit_cond, &submit_gate);
    struct timespec deadline; clock_gettime(CLOCK_REALTIME, &deadline); deadline.tv_sec++;
    int waited = 0;
    while (!reused_done && waited != ETIMEDOUT)
        waited = pthread_cond_timedwait(&submit_cond, &submit_gate, &deadline);
    block_destroy = 0; pthread_cond_broadcast(&submit_cond);
    pthread_mutex_unlock(&submit_gate);
    REQUIRE(pthread_join(destroying, NULL) == 0 && pthread_join(submitting, NULL) == 0);
    REQUIRE(cuGraphLaunch(y, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(used == 656 * MIB);
    gpu_done = 1; destroy(a, x); destroy(b, y); clean();
    cuda_library_entry[OVERRIDE_cuStreamGetId].fn_ptr = native_id;
}

/** Verify graph submission admission, completion, and synchronization interleavings. */
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
    DRIVER(cuThreadExchangeStreamCaptureMode, driver_exchange_mode);
    DRIVER(cuDeviceGraphMemTrim, driver_trim);
    DRIVER(cuStreamGetId, driver_stream_id);
    DRIVER(cuStreamGetCtx, driver_stream_ctx);
    DRIVER(cuCtxGetCurrent, driver_current);
    DRIVER(cuCtxGetDevice, driver_device);
    DRIVER(cuDeviceGetCount, driver_device_count);
    DRIVER(cuCtxSynchronize, driver_context_sync);
    DRIVER(cuStreamSynchronize, driver_stream_sync);
    DRIVER(cuStreamDestroy_v2, driver_stream_destroy);
    DRIVER(cuCtxGetId, driver_context_id);
    DRIVER(cuCtxPushCurrent_v2, driver_push);
    DRIVER(cuCtxPopCurrent_v2, driver_pop);
    DRIVER(cuStreamGetCaptureInfo_v3, driver_capture);
    DRIVER(cuStreamBeginCapture_v2, driver_begin_capture);
    DRIVER(cuStreamEndCapture, driver_end_capture);
    test_external_capture_queries();
    test_capture_context_failure(); test_event_failure(); test_serial_and_overlap();
    for (int mode = 0; mode < 3; mode++) test_persistent(mode);
    test_submission_failure(); test_persistent_replay_after_free(); test_destroy_inflight();
    test_context_lifetime();
#if CUDA_VERSION >= 12040
    DRIVER(cuCtxFromGreenCtx, driver_green_context);
    DRIVER(cuGreenCtxDestroy, driver_green_destroy);
    test_green_context_lifetime();
#endif
    test_reused_context_handle(); test_record_failure_sync();
    test_stream_fallback(); test_stream_destroy_race(); test_cached_replay(); test_host_submission_lifetime();
    for (int context = 0; context < 2; context++)
        for (int pending = 0; pending < 2; pending++) test_sync_submission_race(context, pending);
    puts("PASS: graph overlap, persistent admission, event lifetime, and concurrent synchronization");
    return 0;
}

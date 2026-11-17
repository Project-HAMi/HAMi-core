// Copyright 2026 The HAMi Authors.
#include <stdio.h>

#include "test/graph_accounting_fixture.h"

/** Keep serial graph reuse while charging retained pool backing. */
static void test_serial_pool(void) {
    CUgraph a = make_graph(), b = make_graph();
    REQUIRE(add(a, 300, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(add(b, 300, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec x = instantiate(a), y = instantiate(b);
    REQUIRE(cuGraphLaunch(x, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(cuGraphLaunch(y, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(used == 556 * MIB);
    int queries = pool_queries;
    for (int i = 0; i < 1000; i++) REQUIRE(cuGraphLaunch(y, (CUstream)10) == CUDA_SUCCESS);
    REQUIRE(pool_queries == queries);
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_SUCCESS);
    release_device_memory(0, 300 * MIB);
    reserved = 640 * MIB;
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(used == 896 * MIB);
    destroy(a, x); destroy(b, y);
    REQUIRE(used == 896 * MIB);
    clean();
}

/** Preserve output metadata so replay restores the allocation charge after an external free. */
static void test_output_replay(void) {
    CUgraph graph = make_graph(); CUdeviceptr ptr;
    REQUIRE(add(graph, 512, 0, &ptr) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph);
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    reserved = 512 * MIB;
    graph_ptr_freed(ptr);
    REQUIRE(used == 768 * MIB);
    gpu_done = 1;
    REQUIRE(cuDeviceGraphMemTrim(0) == CUDA_SUCCESS);
    REQUIRE(used == 256 * MIB);
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(used == 768 * MIB);
    destroy(graph, exec); graph_ptr_freed(ptr);
    clean();
}

/** Recompute a cached peak after its largest graph is removed. */
static void test_peak_removal(void) {
    CUgraph a = make_graph(), b = make_graph(), c = make_graph();
    REQUIRE(add(a, 300, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(add(b, 500, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(add(c, 400, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(used == 756 * MIB);
    REQUIRE(cuGraphDestroy(b) == CUDA_SUCCESS);
    REQUIRE(used == 656 * MIB);
    REQUIRE(cuGraphDestroy(c) == CUDA_SUCCESS);
    REQUIRE(used == 556 * MIB);
    REQUIRE(cuGraphDestroy(a) == CUDA_SUCCESS);
    clean();
}

/** Keep a previous charge when the driver cannot confirm that pool backing was freed. */
static void test_query_failure(void) {
    CUgraph graph = make_graph();
    REQUIRE(add(graph, 300, 1, NULL) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph);
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    reserved = 640 * MIB;
    REQUIRE(oom_check(0, 0) == 0 && used == 896 * MIB);
    pool_result = CUDA_ERROR_INVALID_VALUE;
    gpu_done = 1;
    destroy(graph, exec);
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(used == 896 * MIB);
    pool_result = CUDA_SUCCESS;
    clean();
}

/** Protect allocation generations when a graph handle is reused. */
static void test_graph_handle_reuse(void) {
    CUgraph graph = make_graph(); CUdeviceptr old_ptr;
    REQUIRE(add(graph, 400, 0, &old_ptr) == CUDA_SUCCESS);
    CUgraphExec old_exec = instantiate(graph);
    REQUIRE(cuGraphLaunch(old_exec, NULL) == CUDA_SUCCESS);
    destroy(graph, old_exec);
    REQUIRE(add(graph, 300, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(used == 956 * MIB);
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    graph_ptr_freed(old_ptr);
    clean();
}

/** Suppress pool queries while graph allocations are captured. */
static void test_capture_queries(void) {
    CUgraph graph;
    REQUIRE(cuStreamBeginCapture_v2((CUstream)10, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS);
    int before = pool_queries;
    REQUIRE(add(capturing, 100, 0, NULL) == CUDA_SUCCESS);
    REQUIRE(oom_check(0, 0) == 0 && pool_queries == before);
    REQUIRE(cuStreamEndCapture((CUstream)10, &graph) == CUDA_SUCCESS);
    REQUIRE(pool_queries > before);
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
    clean();
}

/** Include live outputs when admitting another graph's pending allocation demand. */
static void test_persistent_and_pending(void) {
    CUgraph a = make_graph(), b = make_graph(); CUdeviceptr ptr;
    REQUIRE(add(a, 400, 0, &ptr) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(a);
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    int before = nodes;
    REQUIRE(add(b, 400, 1, NULL) == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(nodes == before && used == 656 * MIB);
    REQUIRE(add(b, 300, 1, NULL) == CUDA_SUCCESS);
    REQUIRE(used == 956 * MIB);
    REQUIRE(cuGraphDestroy(b) == CUDA_SUCCESS);
    destroy(a, exec); graph_ptr_freed(ptr); clean();
}

/** Reject a second live output before Launch, Upload, or parameter-based upload. */
static void test_output_admission(int mode) {
    CUgraph a = make_graph(), b = make_graph(); CUdeviceptr p, q;
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
        CUresult (*submit)(CUgraphExec, CUstream) = mode ? cuGraphUpload : cuGraphLaunch;
        REQUIRE(submit(x, NULL) == CUDA_SUCCESS);
        int before = submissions;
        REQUIRE(submit(y, NULL) == CUDA_ERROR_OUT_OF_MEMORY);
        REQUIRE(submissions == before);
        destroy(b, y);
    }
    REQUIRE(used == 768 * MIB);
    destroy(a, x); graph_ptr_freed(p); graph_ptr_freed(q); clean();
}

/** Restore the prospective charge when the native submission fails. */
static void test_output_submission_failure(void) {
    CUgraph a = make_graph(), b = make_graph(); CUdeviceptr p, q;
    REQUIRE(add(a, 400, 0, &p) == CUDA_SUCCESS);
    REQUIRE(add(b, 300, 0, &q) == CUDA_SUCCESS);
    CUgraphExec x = instantiate(a), y = instantiate(b);
    REQUIRE(cuGraphLaunch(x, NULL) == CUDA_SUCCESS);
    submit_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(cuGraphLaunch(y, NULL) == submit_result);
    submit_result = CUDA_SUCCESS;
    REQUIRE(used == 656 * MIB);
    destroy(a, x); destroy(b, y); graph_ptr_freed(p); graph_ptr_freed(q); clean();
}

/** Return from a deliberately paused native output submission. */
static void *submit_output(void *exec) {
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    return NULL;
}

/** Keep a pending output charged while its graph and executable are destroyed. */
static void test_output_host_lifetime(void) {
    CUgraph graph = make_graph(); CUdeviceptr ptr;
    REQUIRE(add(graph, 600, 0, &ptr) == CUDA_SUCCESS);
    CUgraphExec exec = instantiate(graph); pthread_t thread;
    block_submit = 1; submit_entered = 0;
    REQUIRE(pthread_create(&thread, NULL, submit_output, exec) == 0);
    pthread_mutex_lock(&submit_gate);
    while (!submit_entered) pthread_cond_wait(&submit_cond, &submit_gate);
    pthread_mutex_unlock(&submit_gate);
    destroy(graph, exec);
    REQUIRE(reserve_device_memory(0, 300 * MIB) == CUDA_ERROR_OUT_OF_MEMORY);
    pthread_mutex_lock(&submit_gate);
    block_submit = 0; pthread_cond_broadcast(&submit_cond);
    pthread_mutex_unlock(&submit_gate);
    REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(used == 856 * MIB);
    graph_ptr_freed(ptr); clean();
}

/** Preserve cached backing across rebuilds and trim it only for a refused reservation. */
static void test_trim_pressure(void) {
    int before = trim_calls;
    for (int i = 0; i < 8; i++) {
        CUgraph graph = make_graph();
        REQUIRE(add(graph, 300, 1, NULL) == CUDA_SUCCESS);
        CUgraphExec exec = instantiate(graph);
        REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
        reserved = 300 * MIB; gpu_done = 1;
        destroy(graph, exec);
        REQUIRE(used == 556 * MIB && trim_calls == before);
    }
    REQUIRE(reserve_device_memory(0, 600 * MIB) == CUDA_SUCCESS);
    REQUIRE(trim_calls == before + 1 && used == 856 * MIB);
    release_device_memory(0, 600 * MIB); clean();
}

/** Exercise pooled quota and pointer lifetimes through production CUDA wrappers. */
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
    DRIVER(cuDeviceGetGraphMemAttribute, driver_reserved);
    DRIVER(cuThreadExchangeStreamCaptureMode, driver_exchange_mode);
    DRIVER(cuDeviceGraphMemTrim, driver_trim);
    DRIVER(cuStreamGetCaptureInfo_v3, driver_capture);
    DRIVER(cuStreamBeginCapture_v2, driver_begin_capture);
    DRIVER(cuStreamEndCapture, driver_end_capture);
    DRIVER(cuCtxGetDevice, driver_device);
    DRIVER(cuDeviceGetCount, driver_device_count);
    test_persistent_and_pending(); test_trim_pressure();
    for (int mode = 0; mode < 3; mode++) test_output_admission(mode);
    test_output_submission_failure(); test_output_host_lifetime();
    test_serial_pool(); test_output_replay(); test_peak_removal();
    test_query_failure(); test_graph_handle_reuse(); test_capture_queries();
    puts("PASS: graph pool reuse, retained backing, cached peaks, and output lifetimes");
    return 0;
}

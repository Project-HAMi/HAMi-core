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
    DRIVER(cuGraphLaunch, driver_submit);
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
    test_serial_pool(); test_output_replay(); test_peak_removal();
    test_query_failure(); test_graph_handle_reuse(); test_capture_queries();
    puts("PASS: graph pool reuse, retained backing, cached peaks, and output lifetimes");
    return 0;
}

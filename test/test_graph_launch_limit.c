// Copyright 2026 The HAMi Authors.
/* A CUDA graph launch must pass the compute limiter like the kernels inside it would
 * one by one: each launch is charged the summed grid x block of its kernel nodes. */
#include <stdio.h>

#include "test/graph_accounting_fixture.h"

/** Add a kernel node with the given launch size to a simulated graph. */
static void add_kernel(CUgraph graph, unsigned int grid, unsigned int block) {
    graph_nodes[++serial].graph = graph;
    graph_nodes[serial].type = CU_GRAPH_NODE_TYPE_KERNEL;
    graph_nodes[serial].grid = grid;
    graph_nodes[serial].block = block;
}

/** Nest a child graph inside a simulated graph. */
static void add_child(CUgraph graph, CUgraph child) {
    graph_nodes[++serial].graph = graph;
    graph_nodes[serial].type = CU_GRAPH_NODE_TYPE_GRAPH;
    graph_nodes[serial].child = child;
}

static void reset_limits(void) {
    limited_threads = 0;
    limited_launches = 0;
}

/** Kernel nodes of the graph and of its child graphs are summed and charged per launch. */
static void test_launch_charges_kernel_threads(void) {
    CUgraph child = make_graph();
    add_kernel(child, 2, 32);
    CUgraph graph = make_graph();
    add_kernel(graph, 4, 128);
    add_kernel(graph, 3, 64);
    add_child(graph, child);
    CUgraphExec exec = instantiate(graph);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 1 && limited_threads == 4 * 128 + 3 * 64 + 2 * 32);
    REQUIRE(cuGraphLaunch_ptsz(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 2 && limited_threads == 2 * (4 * 128 + 3 * 64 + 2 * 32));
    REQUIRE(cuGraphUpload(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 2);
    destroy(graph, exec);
}

/** A graph without kernels, or a process the limiter does not track, is not charged. */
static void test_uncharged_launches(void) {
    CUgraph empty = make_graph();
    CUgraphExec exec = instantiate(empty);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 0);
    destroy(empty, exec);

    CUgraph graph = make_graph();
    add_kernel(graph, 8, 256);
    exec = instantiate(graph);
    pidfound = 0;
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 0);
    pidfound = 1;
    destroy(graph, exec);
}

/** Destroying executable graphs keeps the remaining ones charged their own cost. */
static void test_destroy_keeps_other_costs(void) {
    enum { N = 2000 };
    static CUgraph graphs[N];
    static CUgraphExec execs[N];
    for (int i = 0; i < N; i++) {
        graphs[i] = make_graph();
        add_kernel(graphs[i], (unsigned int)i + 1, 1);
        execs[i] = instantiate(graphs[i]);
    }
    for (int i = 0; i < N; i += 2) destroy(graphs[i], execs[i]);
    for (int i = 1; i < N; i += 2) {
        reset_limits();
        REQUIRE(cuGraphLaunch(execs[i], NULL) == CUDA_SUCCESS);
        REQUIRE(limited_launches == 1 && limited_threads == i + 1);
    }
    for (int i = 1; i < N; i += 2) destroy(graphs[i], execs[i]);
}

#if CUDA_VERSION >= 12030
/** A conditional node is charged its largest body, since at most one body runs per evaluation. */
static void test_conditional_bodies(void) {
    CUgraph graph = make_graph();
    add_kernel(graph, 4, 32);
    CUgraphNodeParams params = {0};
    params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    params.conditional.type = CU_GRAPH_COND_TYPE_IF;
    params.conditional.size = 2;
    CUgraphNode node;
    REQUIRE(cuGraphAddNode_v2(&node, graph, NULL, NULL, 0, &params) == CUDA_SUCCESS);
    CUgraph then_body = params.conditional.phGraph_out[0];
    CUgraph else_body = params.conditional.phGraph_out[1];
    add_kernel(then_body, 8, 64);
    add_kernel(else_body, 2, 64);
    CUgraphExec exec = instantiate(graph);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 1 && limited_threads == 4 * 32 + 8 * 64);
    destroy(graph, exec);
    graph = make_graph();
    add_kernel(graph, 4, 32);
    exec = instantiate(graph);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 4 * 32);
    destroy(graph, exec);
}

/** With cuGraphNodeGetParams the driver's bodies are used, so a conditional node that was not
 * recorded at creation is still charged; if the call fails the recorded bodies are used. */
static void test_conditional_bodies_from_driver(void) {
    static CUgraph bodies[2];
    bodies[0] = make_graph();
    bodies[1] = make_graph();
    add_kernel(bodies[0], 8, 64);
    add_kernel(bodies[1], 2, 64);
    CUgraph unrecorded = make_graph();
    add_kernel(unrecorded, 4, 32);
    graph_nodes[++serial].graph = unrecorded;
    graph_nodes[serial].type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    graph_nodes[serial].bodies = bodies;
    graph_nodes[serial].size = 2;
    CUgraphExec exec = instantiate(unrecorded);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 4 * 32);
    REQUIRE(cuGraphExecDestroy(exec) == CUDA_SUCCESS);

    cuda_library_entry[OVERRIDE_cuGraphNodeGetParams].fn_ptr = (void *)driver_node_get_params;
    exec = instantiate(unrecorded);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 4 * 32 + 8 * 64);
    REQUIRE(cuGraphExecDestroy(exec) == CUDA_SUCCESS);

    test_conditional_bodies();
    node_params_result = CUDA_ERROR_NOT_SUPPORTED;
    test_conditional_bodies();
    node_params_result = CUDA_SUCCESS;
    cuda_library_entry[OVERRIDE_cuGraphNodeGetParams].fn_ptr = NULL;
    REQUIRE(cuGraphDestroy(unrecorded) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(bodies[0]) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(bodies[1]) == CUDA_SUCCESS);
}
#endif

/** An executable graph updated in place is charged for its new graph; a failed update keeps the old cost. */
static void test_exec_update(void) {
    CUgraph first = make_graph();
    add_kernel(first, 4, 64);
    CUgraphExec exec = instantiate(first);
    CUgraph second = make_graph();
    add_kernel(second, 16, 64);
    CUgraphExecUpdateResultInfo info;
    REQUIRE(cuGraphExecUpdate_v2(exec, second, &info) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 16 * 64);
    CUgraph third = make_graph();
    add_kernel(third, 8, 32);
    CUgraphExecUpdateResult result;
    CUgraphNode error_node;
    REQUIRE(cuGraphExecUpdate(exec, third, &error_node, &result) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 1 && limited_threads == 8 * 32);
    /* The driver rejects an update whose node count differs, so the exec keeps its kernel. */
    CUgraph empty = make_graph();
    update_result = CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
    REQUIRE(cuGraphExecUpdate_v2(exec, empty, &info) == CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE);
    REQUIRE(cuGraphExecUpdate(exec, empty, &error_node, &result) == CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE);
    update_result = CUDA_SUCCESS;
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 1 && limited_threads == 8 * 32);
    REQUIRE(cuGraphExecDestroy(exec) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(first) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(second) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(third) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(empty) == CUDA_SUCCESS);
}

/** Changing one kernel node of an exec recharges that node only; failed or unknown changes keep the cost. */
static void test_exec_kernel_set_params(void) {
    CUgraph child = make_graph();
    add_kernel(child, 2, 32);
    CUgraph graph = make_graph();
    add_kernel(graph, 4, 128);
    CUgraphNode resized = (CUgraphNode)serial;
    add_kernel(graph, 3, 64);
    add_child(graph, child);
    CUgraphExec exec = instantiate(graph);

    CUDA_KERNEL_NODE_PARAMS p = {.gridDimX = 10, .gridDimY = 2, .gridDimZ = 1,
                                 .blockDimX = 32, .blockDimY = 1, .blockDimZ = 1};
    REQUIRE(cuGraphExecKernelNodeSetParams_v2(exec, resized, &p) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 20 * 32 + 3 * 64 + 2 * 32);

    CUDA_KERNEL_NODE_PARAMS_v1 legacy = {.gridDimX = 1, .gridDimY = 1, .gridDimZ = 1,
                                         .blockDimX = 16, .blockDimY = 1, .blockDimZ = 1};
    REQUIRE(cuGraphExecKernelNodeSetParams(exec, resized, &legacy) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 16 + 3 * 64 + 2 * 32);

    CUgraphNodeParams generic = {0};
    generic.type = CU_GRAPH_NODE_TYPE_KERNEL;
    generic.kernel.gridDimX = 5; generic.kernel.gridDimY = 1; generic.kernel.gridDimZ = 1;
    generic.kernel.blockDimX = 64; generic.kernel.blockDimY = 1; generic.kernel.blockDimZ = 1;
    REQUIRE(cuGraphExecNodeSetParams(exec, resized, &generic) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 5 * 64 + 3 * 64 + 2 * 32);

    set_params_result = CUDA_ERROR_INVALID_VALUE;
    p.gridDimX = 1000;
    REQUIRE(cuGraphExecKernelNodeSetParams_v2(exec, resized, &p) == CUDA_ERROR_INVALID_VALUE);
    set_params_result = CUDA_SUCCESS;
    REQUIRE(cuGraphExecKernelNodeSetParams_v2(exec, (CUgraphNode)7, &p) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 5 * 64 + 3 * 64 + 2 * 32);
    destroy(graph, exec);
    REQUIRE(cuGraphDestroy(child) == CUDA_SUCCESS);
}

/** After a whole-graph update, setters still name nodes of the instantiation graph and recharge the updated cost. */
static void test_set_params_after_update(void) {
    CUgraph first = make_graph();
    add_kernel(first, 1, 64);
    CUgraphNode a = (CUgraphNode)serial;
    add_kernel(first, 2, 64);
    CUgraphNode b = (CUgraphNode)serial;
    CUgraphExec exec = instantiate(first);
    CUgraph second = make_graph();
    add_kernel(second, 4, 64);
    add_kernel(second, 8, 64);
    CUgraphExecUpdateResultInfo info;
    REQUIRE(cuGraphExecUpdate_v2(exec, second, &info) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 4 * 64 + 8 * 64);

    CUDA_KERNEL_NODE_PARAMS p = {.gridDimX = 64, .gridDimY = 1, .gridDimZ = 1,
                                 .blockDimX = 64, .blockDimY = 1, .blockDimZ = 1};
    REQUIRE(cuGraphExecKernelNodeSetParams_v2(exec, a, &p) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 64 * 64 + 8 * 64);

    CUgraph third = make_graph();
    add_kernel(third, 3, 64);
    add_kernel(third, 5, 64);
    REQUIRE(cuGraphExecUpdate_v2(exec, third, &info) == CUDA_SUCCESS);
    REQUIRE(cuGraphExecKernelNodeSetParams_v2(exec, b, &p) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 3 * 64 + 64 * 64);
    REQUIRE(cuGraphExecDestroy(exec) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(first) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(second) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(third) == CUDA_SUCCESS);
}

/** Replacing a child graph of an exec recharges that node with the new child's kernels. */
static void test_exec_child_set_params(void) {
    CUgraph child = make_graph();
    add_kernel(child, 1, 32);
    CUgraph graph = make_graph();
    add_kernel(graph, 4, 64);
    add_child(graph, child);
    CUgraphNode child_node = (CUgraphNode)serial;
    CUgraphExec exec = instantiate(graph);

    CUgraph bigger = make_graph();
    add_kernel(bigger, 16, 256);
    CUgraphNodeParams generic = {0};
    generic.type = CU_GRAPH_NODE_TYPE_GRAPH;
    generic.graph.graph = bigger;
    REQUIRE(cuGraphExecNodeSetParams(exec, child_node, &generic) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 4 * 64 + 16 * 256);

    CUgraph smaller = make_graph();
    add_kernel(smaller, 2, 64);
    REQUIRE(cuGraphExecChildGraphNodeSetParams(exec, child_node, smaller) == CUDA_SUCCESS);
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 4 * 64 + 2 * 64);

    set_params_result = CUDA_ERROR_INVALID_VALUE;
    REQUIRE(cuGraphExecChildGraphNodeSetParams(exec, child_node, bigger) == CUDA_ERROR_INVALID_VALUE);
    REQUIRE(cuGraphExecNodeSetParams(exec, child_node, &generic) == CUDA_ERROR_INVALID_VALUE);
    set_params_result = CUDA_SUCCESS;
    reset_limits();
    REQUIRE(cuGraphLaunch(exec, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_threads == 4 * 64 + 2 * 64);
    destroy(graph, exec);
    REQUIRE(cuGraphDestroy(child) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(bigger) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(smaller) == CUDA_SUCCESS);
}

static CUresult driver_launch_kernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                                     unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                                     CUstream stream, void **params, void **extra) {
    (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)shared; (void)stream; (void)params; (void)extra;
    return CUDA_SUCCESS;
}

/** A kernel recorded into a capturing stream is not charged; the graph launch charges it. */
static void test_capture_not_charged(void) {
    reset_limits();
    REQUIRE(cuLaunchKernel((CUfunction)1, 4, 1, 1, 32, 1, 1, 0, NULL, NULL, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 1 && limited_threads == 4 * 32);
    capturing = make_graph();
    reset_limits();
    REQUIRE(cuLaunchKernel((CUfunction)1, 4, 1, 1, 32, 1, 1, 0, NULL, NULL, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 0);
    capturing = NULL;
}

/** A full cost table leaves new graphs uncharged, and destroying an exec still returns. */
static void test_full_table(void) {
    enum { SLOTS = 65536 };
    static CUgraphExec execs[SLOTS];
    CUgraph graph = make_graph();
    add_kernel(graph, 2, 16);
    for (int i = 0; i < SLOTS - 1; i++) execs[i] = instantiate(graph);
    CUgraphExec overflow = instantiate(graph);
    reset_limits();
    REQUIRE(cuGraphLaunch(overflow, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 0);
    REQUIRE(cuGraphLaunch(execs[0], NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 1 && limited_threads == 2 * 16);
    REQUIRE(cuGraphExecDestroy(execs[0]) == CUDA_SUCCESS);
    CUgraphExec fresh = instantiate(graph);
    reset_limits();
    REQUIRE(cuGraphLaunch(fresh, NULL) == CUDA_SUCCESS);
    REQUIRE(limited_launches == 1 && limited_threads == 2 * 16);
    REQUIRE(cuGraphExecDestroy(fresh) == CUDA_SUCCESS);
    REQUIRE(cuGraphExecDestroy(overflow) == CUDA_SUCCESS);
    for (int i = 1; i < SLOTS - 1; i++) REQUIRE(cuGraphExecDestroy(execs[i]) == CUDA_SUCCESS);
    REQUIRE(cuGraphDestroy(graph) == CUDA_SUCCESS);
}

int main(void) {
    log_utils_init();
    setup_graph_queries();
#define DRIVER(name, fn) cuda_library_entry[CUDA_OVERRIDE_ENUM(name)].fn_ptr = (void *)(fn)
    DRIVER(cuGraphInstantiateWithFlags, driver_instantiate);
    DRIVER(cuGraphLaunch, driver_submit);
    DRIVER(cuGraphUpload, driver_submit);
    DRIVER(cuGraphDestroy, driver_destroy);
    DRIVER(cuGraphExecDestroy, driver_destroy);
    DRIVER(cuGraphExecUpdate_v2, driver_exec_update);
    DRIVER(cuGraphExecUpdate, driver_exec_update_legacy);
    DRIVER(cuGraphExecKernelNodeSetParams_v2, driver_exec_kernel_params);
    DRIVER(cuGraphExecKernelNodeSetParams, driver_exec_kernel_params);
    DRIVER(cuGraphExecNodeSetParams, driver_exec_node_params);
    DRIVER(cuGraphExecChildGraphNodeSetParams, driver_exec_child_params);
    DRIVER(cuStreamGetCaptureInfo_v3, driver_capture);
    DRIVER(cuLaunchKernel, driver_launch_kernel);
    DRIVER(cuGraphAddNode_v2, driver_generic_node);
#undef DRIVER
    pidfound = 1;
    test_launch_charges_kernel_threads();
    test_uncharged_launches();
    test_destroy_keeps_other_costs();
#if CUDA_VERSION >= 12030
    test_conditional_bodies();
    test_conditional_bodies_from_driver();
#endif
    test_exec_update();
    test_exec_kernel_set_params();
    test_exec_child_set_params();
    test_set_params_after_update();
    test_capture_not_charged();
    test_full_table();
    printf("graph launch limit: ok\n");
    return 0;
}

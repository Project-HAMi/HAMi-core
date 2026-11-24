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

int main(void) {
    log_utils_init();
    setup_graph_queries();
#define DRIVER(name, fn) cuda_library_entry[CUDA_OVERRIDE_ENUM(name)].fn_ptr = (void *)(fn)
    DRIVER(cuGraphInstantiateWithFlags, driver_instantiate);
    DRIVER(cuGraphLaunch, driver_submit);
    DRIVER(cuGraphUpload, driver_submit);
    DRIVER(cuGraphDestroy, driver_destroy);
    DRIVER(cuGraphExecDestroy, driver_destroy);
#undef DRIVER
    pidfound = 1;
    test_launch_charges_kernel_threads();
    test_uncharged_launches();
    test_destroy_keeps_other_costs();
    printf("graph launch limit: ok\n");
    return 0;
}

// Copyright 2026 The HAMi Authors.
#pragma once
#include <stdio.h>
#include <string.h>

void load_cuda_libraries(void);
static int graph_loader_handle, graph_loader_v2_present;

void *__wrap_dlopen(const char *name, int flags) {
    (void)name; (void)flags; return &graph_loader_handle;
}

int __wrap_dlclose(void *handle) { (void)handle; return 0; }

static CUresult graph_loader_legacy(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                                     size_t count, CUgraphNodeParams *params) {
    (void)node; (void)graph; (void)deps; (void)count; (void)params; return CUDA_SUCCESS;
}

static CUresult graph_loader_v2(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                                 const CUgraphEdgeData *data, size_t count, CUgraphNodeParams *params) {
    (void)data; return graph_loader_legacy(node, graph, deps, count, params);
}

static CUresult graph_loader_init(unsigned int flags) { (void)flags; return CUDA_SUCCESS; }

/** Expose only the original five-argument generic-node ABI to the production loader. */
static void *graph_loader_symbol(void *handle, const char *name) {
    (void)handle;
    void *symbol = NULL;
    if (strcmp(name, "cuGraphAddNode") == 0) symbol = graph_loader_legacy;
    if (strcmp(name, "cuInit") == 0) symbol = graph_loader_init;
    if (graph_loader_v2_present && strcmp(name, "cuGraphAddNode_v2") == 0) symbol = graph_loader_v2;
    return symbol;
}

/** Keep an unavailable six-argument ABI null rather than borrowing its incompatible predecessor. */
static int check_graph_loader_abi(void) {
    real_dlsym = graph_loader_symbol;
    load_cuda_libraries();
    void *legacy = graph_loader_legacy;
    if (cuda_library_entry[OVERRIDE_cuGraphAddNode].fn_ptr != legacy ||
        cuda_library_entry[OVERRIDE_cuGraphAddNode_v2].fn_ptr != NULL ||
        cuda_library_entry[OVERRIDE_cuGraphAddNode_v3].fn_ptr != NULL) {
        puts("FAIL generic node loader substituted an incompatible ABI");
        return 1;
    }
    graph_loader_v2_present = 1;
    load_cuda_libraries();
    void *edge_abi = graph_loader_v2;
    if (cuda_library_entry[OVERRIDE_cuGraphAddNode_v2].fn_ptr != edge_abi ||
        cuda_library_entry[OVERRIDE_cuGraphAddNode_v3].fn_ptr != NULL) {
        puts("FAIL generic node loader changed v3 native semantics");
        return 1;
    }
    return 0;
}

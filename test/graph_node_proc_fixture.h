// Copyright 2026 The HAMi Authors.
#pragma once
#include <stdio.h>

CUresult _cuGetProcAddress(const char *symbol, void **pfn, int version, cuuint64_t flags);
static CUresult graph_proc_result;
static CUdriverProcAddressQueryResult graph_proc_status;
static int graph_proc_pointer;
static char graph_native_marker[] = "native-graph-node";

/** Return the configured native generic-node lookup result. */
static CUresult graph_proc_native(const char *symbol, void **pfn, int version, cuuint64_t flags) {
    (void)symbol; (void)version; (void)flags;
    *pfn = graph_proc_pointer ? graph_native_marker : NULL;
    return graph_proc_result;
}

/** Return the configured generic-node query result and version status. */
static CUresult graph_proc_native_v2(const char *symbol, void **pfn, int version, cuuint64_t flags,
                                      CUdriverProcAddressQueryResult *status) {
    CUresult res = graph_proc_native(symbol, pfn, version, flags);
    *status = graph_proc_status;
    return res;
}

/** Check all generic-node ABIs and native outcomes through all procedure lookup routes. */
static int check_graph_node_procs(void) {
    const int versions[] = {12020, 12029, 12030, 13000, 13030, 13039, 13040, 13050, 13051, 99999};
    char *names[] = {"cuGraphAddNode", "cuGraphAddNode_v2", "cuGraphAddNode_v3"};
    g_present[0] = names[0]; g_present[1] = names[1]; g_present[2] = names[2]; g_present[3] = NULL;
    cuda_library_entry[OVERRIDE_cuGetProcAddress].fn_ptr = graph_proc_native;
    cuda_library_entry[OVERRIDE_cuGetProcAddress_v2].fn_ptr = graph_proc_native_v2;
    cuda_library_entry[OVERRIDE_cuGraphAddNode_v3].fn_ptr = graph_native_marker;
    for (int n = 0; n < 3; n++) for (int v = 0; v < 10; v++) for (int pt = 0; pt < 2; pt++)
        for (int entry = 0; entry < 4; entry++) for (int state = 0; state < 4; state++) {
            graph_proc_result = state == 1 ? CUDA_ERROR_NOT_FOUND : CUDA_SUCCESS;
            graph_proc_pointer = state != 2;
            graph_proc_status = state == 3 ? CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT : CU_GET_PROC_ADDRESS_SUCCESS;
            void *pfn = NULL;
            CUdriverProcAddressQueryResult status = STATUS_POISON;
            cuuint64_t flags = pt ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM : CU_GET_PROC_ADDRESS_LEGACY_STREAM;
            CUresult res;
            if (entry == 0) res = cuGetProcAddress(names[n], &pfn, versions[v], flags);
            else if (entry == 1) res = _cuGetProcAddress(names[n], &pfn, versions[v], flags);
            else if (entry == 2) res = cuGetProcAddress_v2(names[n], &pfn, versions[v], flags, &status);
            else res = _cuGetProcAddress_v2(names[n], &pfn, versions[v], flags, &status);
            void *expected = graph_proc_pointer ? graph_native_marker : NULL;
            if (graph_proc_result == CUDA_SUCCESS && graph_proc_pointer && (entry < 2 || state != 3))
                expected = names[n ? n : (versions[v] >= 13040 ? 2 : versions[v] >= 12030)];
            if (res != graph_proc_result || pfn != expected || (entry >= 2 && status != graph_proc_status)) {
                printf("FAIL graph node lookup: %s version=%d entry=%d state=%d\n",
                       names[n], versions[v], entry, state);
                return 1;
            }
        }
    puts("ok   generic graph node ABIs and native lookup outcomes: 960 cases");
    cuda_library_entry[OVERRIDE_cuGraphAddNode_v3].fn_ptr = NULL;
    graph_proc_result = CUDA_SUCCESS; graph_proc_pointer = 1; graph_proc_status = CU_GET_PROC_ADDRESS_SUCCESS;
    for (int entry = 0; entry < 4; entry++) {
        void *pfn = NULL; CUdriverProcAddressQueryResult status = STATUS_POISON;
        CUresult res;
        if (entry == 0) res = cuGetProcAddress(names[0], &pfn, 13060, 0);
        else if (entry == 1) res = _cuGetProcAddress(names[0], &pfn, 13060, 0);
        else if (entry == 2) res = cuGetProcAddress_v2(names[0], &pfn, 13060, 0, &status);
        else res = _cuGetProcAddress_v2(names[0], &pfn, 13060, 0, &status);
        if (res != CUDA_ERROR_NOT_SUPPORTED || pfn != NULL ||
            (entry >= 2 && status != CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT)) return 1;
    }
    return 0;
}

/** Preserve native Trim availability and query outcomes through every lookup route. */
static int check_graph_pool_procs(void) {
    char *name = "cuDeviceGraphMemTrim";
    g_present[0] = name; g_present[1] = NULL;
    cuda_library_entry[OVERRIDE_cuGetProcAddress].fn_ptr = graph_proc_native;
    cuda_library_entry[OVERRIDE_cuGetProcAddress_v2].fn_ptr = graph_proc_native_v2;
    for (int entry = 0; entry < 4; entry++) for (int state = 0; state < 4; state++) {
        graph_proc_result = state == 1 ? CUDA_ERROR_NOT_FOUND : CUDA_SUCCESS;
        graph_proc_pointer = state != 2;
        graph_proc_status = state == 3 ? CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT : CU_GET_PROC_ADDRESS_SUCCESS;
        void *pfn = NULL;
        CUdriverProcAddressQueryResult status = STATUS_POISON;
        CUresult res;
        if (entry == 0) res = cuGetProcAddress(name, &pfn, 11400, 0);
        else if (entry == 1) res = _cuGetProcAddress(name, &pfn, 11400, 0);
        else if (entry == 2) res = cuGetProcAddress_v2(name, &pfn, 11400, 0, &status);
        else res = _cuGetProcAddress_v2(name, &pfn, 11400, 0, &status);
        void *expected = graph_proc_pointer ? graph_native_marker : NULL;
        if (graph_proc_result == CUDA_SUCCESS && graph_proc_pointer && (entry < 2 || state != 3)) expected = name;
        if (res != graph_proc_result || pfn != expected || (entry >= 2 && status != graph_proc_status)) return 1;
    }
    return 0;
}

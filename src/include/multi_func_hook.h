//
// Created by lihuang on 2025/4/11.
//
#ifndef HIHOOK_HEADER_H
#define HIHOOK_HEADER_H

#include <stdlib.h>


typedef struct {
  const char *func_name;      // base func name（like "cuGraphAddDependencies"）
  int min_ver;    // adjust to low version
  int max_ver;    // adjust to high version
  const char *real_name;      // the real name（ "cuGraphAddDependencies_v2"）
} CudaFuncMapEntry;

// if multi func, we can add here
// like 12030，means cuda 12.3 ，cuda.h header may give start at version
// all new add func put here
static CudaFuncMapEntry g_func_map[] = {
    {"cuGraphAddKernelNode", 10000, 11999, "cuGraphAddKernelNode"},
    {"cuGraphAddKernelNode", 12000, 99999, "cuGraphAddKernelNode_v2"},

    {"cuGraphKernelNodeGetParams", 10000, 11999, "cuGraphKernelNodeGetParams"},
    {"cuGraphKernelNodeGetParams", 12000, 99999, "cuGraphKernelNodeGetParams_v2"},

    {"cuGraphKernelNodeSetParams", 10000, 11999, "cuGraphKernelNodeSetParams"},
    {"cuGraphKernelNodeSetParams", 12000, 99999, "cuGraphKernelNodeSetParams_v2"},

    // Boundaries follow cudaTypedefs.h: cuCtxCreate is _v2 below 11.4, _v3
    // from 11.4 and _v4 from 12.5; cuMemAdvise and cuMemPrefetchAsync switch
    // to _v2 at 12.2.  find_symbols_in_table() only probes _v3, _v2 and the
    // bare spelling, so it cannot reach _v4 and answers with the wrong
    // signature elsewhere.  Pin each range explicitly.
    {"cuCtxCreate", 10000, 11039, "cuCtxCreate_v2"},
    {"cuCtxCreate", 11040, 12049, "cuCtxCreate_v3"},
    {"cuCtxCreate", 12050, 99999, "cuCtxCreate_v4"},

    {"cuMemAdvise", 10000, 12019, "cuMemAdvise"},
    {"cuMemAdvise", 12020, 99999, "cuMemAdvise_v2"},

    {"cuMemPrefetchAsync", 10000, 12019, "cuMemPrefetchAsync"},
    {"cuMemPrefetchAsync", 12020, 99999, "cuMemPrefetchAsync_v2"}
};


#endif

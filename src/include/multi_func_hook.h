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

    // cuda.h remaps cuCtxCreate to cuCtxCreate_v2 up to CUDA 12.x and to
    // cuCtxCreate_v4 from CUDA 13.0.  find_symbols_in_table() only probes the
    // _v3, _v2 and bare spellings, so it can never reach the _v4 hook, and on
    // 12.x it answers with _v3, whose five-parameter signature does not match
    // the three-parameter cuCtxCreate the caller has.  Pin both explicitly.
    {"cuCtxCreate", 10000, 12999, "cuCtxCreate_v2"},
    {"cuCtxCreate", 13000, 99999, "cuCtxCreate_v4"},

    // Self up to 12.x, _v2 from 13.0, but the _v2 probe runs first.
    {"cuMemAdvise", 10000, 12999, "cuMemAdvise"},
    {"cuMemAdvise", 13000, 99999, "cuMemAdvise_v2"},

    {"cuMemPrefetchAsync", 10000, 12999, "cuMemPrefetchAsync"},
    {"cuMemPrefetchAsync", 13000, 99999, "cuMemPrefetchAsync_v2"}
};


#endif

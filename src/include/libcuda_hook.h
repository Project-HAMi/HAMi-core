#ifndef __LIBCUDA_HOOK_H__
#define __LIBCUDA_HOOK_H__

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <cuda.h>
#include <pthread.h>

#include "include/log_utils.h"
#include "include/hook.h" 


#define FILENAME_MAX 4096

#define CONTEXT_SIZE 104857600

typedef CUresult (*cuda_sym_t)();


#define CUDA_OVERRIDE_CALL(table, sym, ...)                                    \
  ({    \
    LOG_DEBUG("Hijacking %s", #sym);                                           \
    cuda_sym_t _entry = (cuda_sym_t)FIND_ENTRY(table, sym);               \
    _entry(__VA_ARGS__);                                                       \
  })

typedef enum {
    #define X(func) OVERRIDE_##func,
    CUDA_FUNCTIONS(X)
    X(cuMemGetInfo_v2)
    #undef X

    CUDA_ENTRY_END
}cuda_override_enum_t;

extern entry_t cuda_library_entry[];

#endif  //__LIBCUDA_HOOK_H__

#undef cuGetProcAddress
CUresult cuGetProcAddress( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags );
#undef cuGraphInstantiate
CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);


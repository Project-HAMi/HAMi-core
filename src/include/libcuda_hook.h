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
    /* cuInit Part */
    OVERRIDE_ENUM(cuInit),
    /* cuDeivce Part */
    OVERRIDE_ENUM(cuDeviceGetAttribute),
    OVERRIDE_ENUM(cuDeviceGet),
    OVERRIDE_ENUM(cuDeviceGetCount),
    OVERRIDE_ENUM(cuDeviceGetName),
    OVERRIDE_ENUM(cuDeviceCanAccessPeer),
    OVERRIDE_ENUM(cuDeviceGetP2PAttribute),
    OVERRIDE_ENUM(cuDeviceGetByPCIBusId),
    OVERRIDE_ENUM(cuDeviceGetPCIBusId),
    OVERRIDE_ENUM(cuDeviceGetUuid),
    OVERRIDE_ENUM(cuDeviceGetDefaultMemPool),
    OVERRIDE_ENUM(cuDeviceGetLuid),
    OVERRIDE_ENUM(cuDeviceGetMemPool),
    OVERRIDE_ENUM(cuDeviceTotalMem_v2),
    OVERRIDE_ENUM(cuDriverGetVersion),
    OVERRIDE_ENUM(cuDeviceGetTexture1DLinearMaxWidth),
    OVERRIDE_ENUM(cuDeviceSetMemPool),
    OVERRIDE_ENUM(cuFlushGPUDirectRDMAWrites),

    /* cuContext Part */
    OVERRIDE_ENUM(cuDevicePrimaryCtxGetState),
    OVERRIDE_ENUM(cuDevicePrimaryCtxRetain),
    OVERRIDE_ENUM(cuDevicePrimaryCtxSetFlags_v2),
    OVERRIDE_ENUM(cuDevicePrimaryCtxRelease_v2),
    OVERRIDE_ENUM(cuCtxGetDevice),
    OVERRIDE_ENUM(cuCtxCreate_v2),
    OVERRIDE_ENUM(cuCtxCreate_v3),
    OVERRIDE_ENUM(cuCtxDestroy_v2),
    OVERRIDE_ENUM(cuCtxGetApiVersion),
    OVERRIDE_ENUM(cuCtxGetCacheConfig),
    OVERRIDE_ENUM(cuCtxGetCurrent),
    OVERRIDE_ENUM(cuCtxGetFlags),
    OVERRIDE_ENUM(cuCtxGetLimit),
    OVERRIDE_ENUM(cuCtxGetSharedMemConfig),
    OVERRIDE_ENUM(cuCtxGetStreamPriorityRange),
    OVERRIDE_ENUM(cuCtxPopCurrent_v2),
    OVERRIDE_ENUM(cuCtxPushCurrent_v2),
    OVERRIDE_ENUM(cuCtxSetCacheConfig),
    OVERRIDE_ENUM(cuCtxSetCurrent),
    OVERRIDE_ENUM(cuCtxSetLimit),
    OVERRIDE_ENUM(cuCtxSetSharedMemConfig),
    OVERRIDE_ENUM(cuCtxSynchronize),
    //OVERRIDE_ENUM(cuCtxEnablePeerAccess),
    OVERRIDE_ENUM(cuGetExportTable),

    /* cuStream Part */
    OVERRIDE_ENUM(cuStreamCreate),
    OVERRIDE_ENUM(cuStreamDestroy_v2),
    OVERRIDE_ENUM(cuStreamSynchronize),
    /* cuMemory Part */
    OVERRIDE_ENUM(cuArray3DCreate_v2),
    OVERRIDE_ENUM(cuArrayCreate_v2),
    OVERRIDE_ENUM(cuArrayDestroy),
    OVERRIDE_ENUM(cuMemAlloc_v2),
    OVERRIDE_ENUM(cuMemAllocHost_v2),
    OVERRIDE_ENUM(cuMemAllocManaged),
    OVERRIDE_ENUM(cuMemAllocPitch_v2),
    OVERRIDE_ENUM(cuMemFree_v2),
    OVERRIDE_ENUM(cuMemFreeHost),
    OVERRIDE_ENUM(cuMemHostAlloc),
    OVERRIDE_ENUM(cuMemHostRegister_v2),
    OVERRIDE_ENUM(cuMemHostUnregister),
    OVERRIDE_ENUM(cuMemcpyDtoH_v2),
    OVERRIDE_ENUM(cuMemcpyHtoD_v2),
    OVERRIDE_ENUM(cuMipmappedArrayCreate),
    OVERRIDE_ENUM(cuMipmappedArrayDestroy),
    OVERRIDE_ENUM(cuMemGetInfo_v2),
    OVERRIDE_ENUM(cuMemcpy),
    OVERRIDE_ENUM(cuPointerGetAttribute),
    OVERRIDE_ENUM(cuPointerGetAttributes),
    OVERRIDE_ENUM(cuPointerSetAttribute),
    OVERRIDE_ENUM(cuIpcCloseMemHandle),
    OVERRIDE_ENUM(cuIpcGetMemHandle),
    OVERRIDE_ENUM(cuIpcOpenMemHandle_v2),
    OVERRIDE_ENUM(cuMemGetAddressRange_v2),
    OVERRIDE_ENUM(cuMemcpyAsync),
    OVERRIDE_ENUM(cuMemcpyAtoD_v2),
    OVERRIDE_ENUM(cuMemcpyDtoA_v2),
    OVERRIDE_ENUM(cuMemcpyDtoD_v2),
    OVERRIDE_ENUM(cuMemcpyDtoDAsync_v2),
    OVERRIDE_ENUM(cuMemcpyDtoHAsync_v2),
    OVERRIDE_ENUM(cuMemcpyHtoDAsync_v2),
    OVERRIDE_ENUM(cuMemcpyPeer),
    OVERRIDE_ENUM(cuMemcpyPeerAsync),
    OVERRIDE_ENUM(cuMemsetD16_v2),
    OVERRIDE_ENUM(cuMemsetD16Async),
    OVERRIDE_ENUM(cuMemsetD2D16_v2),
    OVERRIDE_ENUM(cuMemsetD2D16Async),
    OVERRIDE_ENUM(cuMemsetD2D32_v2),
    OVERRIDE_ENUM(cuMemsetD2D32Async),
    OVERRIDE_ENUM(cuMemsetD2D8_v2),
    OVERRIDE_ENUM(cuMemsetD2D8Async),
    OVERRIDE_ENUM(cuMemsetD32_v2),
    OVERRIDE_ENUM(cuMemsetD32Async),
    OVERRIDE_ENUM(cuMemsetD8_v2),
    OVERRIDE_ENUM(cuMemsetD8Async),
    OVERRIDE_ENUM(cuMemAdvise),
    OVERRIDE_ENUM(cuFuncSetCacheConfig),
    OVERRIDE_ENUM(cuFuncSetSharedMemConfig),
    OVERRIDE_ENUM(cuFuncGetAttribute),
    OVERRIDE_ENUM(cuFuncSetAttribute),
    OVERRIDE_ENUM(cuLaunchKernel),
    OVERRIDE_ENUM(cuLaunchCooperativeKernel),
    /* cuEvent Part */
    OVERRIDE_ENUM(cuEventCreate),
    OVERRIDE_ENUM(cuEventDestroy_v2),
    OVERRIDE_ENUM(cuModuleLoad),
    OVERRIDE_ENUM(cuModuleLoadData),
    OVERRIDE_ENUM(cuModuleLoadDataEx),
    OVERRIDE_ENUM(cuModuleLoadFatBinary),
    OVERRIDE_ENUM(cuModuleGetFunction),
    OVERRIDE_ENUM(cuModuleUnload),
    OVERRIDE_ENUM(cuModuleGetGlobal_v2),
    OVERRIDE_ENUM(cuModuleGetTexRef),
    OVERRIDE_ENUM(cuModuleGetSurfRef),
    OVERRIDE_ENUM(cuLinkAddData_v2),
    OVERRIDE_ENUM(cuLinkCreate_v2),
    OVERRIDE_ENUM(cuLinkAddFile_v2),
    OVERRIDE_ENUM(cuLinkComplete),
    OVERRIDE_ENUM(cuLinkDestroy),

    /* Virtual Memory Part */
    OVERRIDE_ENUM(cuMemAddressReserve),
    OVERRIDE_ENUM(cuMemCreate),
    OVERRIDE_ENUM(cuMemMap),
    OVERRIDE_ENUM(cuMemAllocAsync),
    OVERRIDE_ENUM(cuMemFreeAsync),
    /* cuda11.7 new api memory part */
    OVERRIDE_ENUM(cuMemHostGetDevicePointer_v2),
    OVERRIDE_ENUM(cuMemHostGetFlags),
    OVERRIDE_ENUM(cuMemPoolTrimTo),
    OVERRIDE_ENUM(cuMemPoolSetAttribute),
    OVERRIDE_ENUM(cuMemPoolGetAttribute),
    OVERRIDE_ENUM(cuMemPoolSetAccess),
    OVERRIDE_ENUM(cuMemPoolGetAccess),
    OVERRIDE_ENUM(cuMemPoolCreate),
    OVERRIDE_ENUM(cuMemPoolDestroy),
    OVERRIDE_ENUM(cuMemAllocFromPoolAsync),
    OVERRIDE_ENUM(cuMemPoolExportToShareableHandle),
    OVERRIDE_ENUM(cuMemPoolImportFromShareableHandle),
    OVERRIDE_ENUM(cuMemPoolExportPointer),
    OVERRIDE_ENUM(cuMemPoolImportPointer),
    OVERRIDE_ENUM(cuMemcpy2DUnaligned_v2),
    OVERRIDE_ENUM(cuMemcpy2DAsync_v2),
    OVERRIDE_ENUM(cuMemcpy3D_v2),
    OVERRIDE_ENUM(cuMemcpy3DAsync_v2),
    OVERRIDE_ENUM(cuMemcpy3DPeer),
    OVERRIDE_ENUM(cuMemcpy3DPeerAsync),
    OVERRIDE_ENUM(cuMemPrefetchAsync),
    OVERRIDE_ENUM(cuMemRangeGetAttribute),
    OVERRIDE_ENUM(cuMemRangeGetAttributes),
    /* cuda 11.7 external resource management */
    OVERRIDE_ENUM(cuImportExternalMemory),
    OVERRIDE_ENUM(cuExternalMemoryGetMappedBuffer),
    OVERRIDE_ENUM(cuExternalMemoryGetMappedMipmappedArray),
    OVERRIDE_ENUM(cuDestroyExternalMemory),
    OVERRIDE_ENUM(cuImportExternalSemaphore),
    OVERRIDE_ENUM(cuSignalExternalSemaphoresAsync),
    OVERRIDE_ENUM(cuWaitExternalSemaphoresAsync),
    OVERRIDE_ENUM(cuDestroyExternalSemaphore),
    /* cuda graph part */
    OVERRIDE_ENUM(cuGraphCreate),
    OVERRIDE_ENUM(cuGraphAddKernelNode_v2),
    OVERRIDE_ENUM(cuGraphKernelNodeGetParams_v2),
    OVERRIDE_ENUM(cuGraphKernelNodeSetParams_v2),
    OVERRIDE_ENUM(cuGraphAddMemcpyNode),
    OVERRIDE_ENUM(cuGraphMemcpyNodeGetParams),
    OVERRIDE_ENUM(cuGraphMemcpyNodeSetParams),
    OVERRIDE_ENUM(cuGraphAddMemsetNode),
    OVERRIDE_ENUM(cuGraphMemsetNodeGetParams),
    OVERRIDE_ENUM(cuGraphMemsetNodeSetParams),
    OVERRIDE_ENUM(cuGraphAddHostNode),
    OVERRIDE_ENUM(cuGraphHostNodeGetParams),
    OVERRIDE_ENUM(cuGraphHostNodeSetParams),
    OVERRIDE_ENUM(cuGraphAddChildGraphNode),
    OVERRIDE_ENUM(cuGraphChildGraphNodeGetGraph),
    OVERRIDE_ENUM(cuGraphAddEmptyNode),
    OVERRIDE_ENUM(cuGraphAddEventRecordNode),
    OVERRIDE_ENUM(cuGraphEventRecordNodeGetEvent),
    OVERRIDE_ENUM(cuGraphEventRecordNodeSetEvent),
    OVERRIDE_ENUM(cuGraphAddEventWaitNode),
    OVERRIDE_ENUM(cuGraphEventWaitNodeGetEvent),
    OVERRIDE_ENUM(cuGraphEventWaitNodeSetEvent),
    OVERRIDE_ENUM(cuGraphAddExternalSemaphoresSignalNode),
    OVERRIDE_ENUM(cuGraphExternalSemaphoresSignalNodeGetParams),
    OVERRIDE_ENUM(cuGraphExternalSemaphoresSignalNodeSetParams),
    OVERRIDE_ENUM(cuGraphAddExternalSemaphoresWaitNode),
    OVERRIDE_ENUM(cuGraphExternalSemaphoresWaitNodeGetParams),
    OVERRIDE_ENUM(cuGraphExternalSemaphoresWaitNodeSetParams),
    OVERRIDE_ENUM(cuGraphExecExternalSemaphoresSignalNodeSetParams),
    OVERRIDE_ENUM(cuGraphExecExternalSemaphoresWaitNodeSetParams),
    OVERRIDE_ENUM(cuGraphClone),
    OVERRIDE_ENUM(cuGraphNodeFindInClone),
    OVERRIDE_ENUM(cuGraphNodeGetType),
    OVERRIDE_ENUM(cuGraphGetNodes),
    OVERRIDE_ENUM(cuGraphGetRootNodes),
    OVERRIDE_ENUM(cuGraphGetEdges),
    OVERRIDE_ENUM(cuGraphNodeGetDependencies),
    OVERRIDE_ENUM(cuGraphNodeGetDependentNodes),
    OVERRIDE_ENUM(cuGraphAddDependencies),
    OVERRIDE_ENUM(cuGraphRemoveDependencies),
    OVERRIDE_ENUM(cuGraphDestroyNode),
    OVERRIDE_ENUM(cuGraphInstantiate),
    OVERRIDE_ENUM(cuGraphInstantiateWithFlags),
    OVERRIDE_ENUM(cuGraphUpload),
    OVERRIDE_ENUM(cuGraphLaunch),
    OVERRIDE_ENUM(cuGraphExecDestroy),
    OVERRIDE_ENUM(cuGraphDestroy),

    OVERRIDE_ENUM(cuGetProcAddress),
    OVERRIDE_ENUM(cuGetProcAddress_v2),
    CUDA_ENTRY_END
}cuda_override_enum_t;

extern entry_t cuda_library_entry[];

#endif  //__LIBCUDA_HOOK_H__

#undef cuGetProcAddress
CUresult cuGetProcAddress( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags );
#undef cuGraphInstantiate
CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);


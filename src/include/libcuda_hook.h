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

typedef struct {
  void *fn_ptr;
  char *name;
} cuda_entry_t;

#define FILENAME_MAX 4096

#define CONTEXT_SIZE 104857600

typedef CUresult (*cuda_sym_t)();

#define CUDA_OVERRIDE_ENUM(x) OVERRIDE_##x

#define CUDA_FIND_ENTRY(table, sym) ({ (table)[CUDA_OVERRIDE_ENUM(sym)].fn_ptr; })

#define CUDA_OVERRIDE_CALL(table, sym, ...)                                    \
  ({    \
    LOG_DEBUG("Hijacking %s", #sym);                                           \
    cuda_sym_t _entry = (cuda_sym_t)CUDA_FIND_ENTRY(table, sym);               \
    _entry(__VA_ARGS__);                                                       \
  })

typedef enum {
    /* cuInit Part */
    CUDA_OVERRIDE_ENUM(cuInit),
    /* cuDeivce Part */
    CUDA_OVERRIDE_ENUM(cuDeviceGetAttribute),
    CUDA_OVERRIDE_ENUM(cuDeviceGet),
    CUDA_OVERRIDE_ENUM(cuDeviceGetCount),
    CUDA_OVERRIDE_ENUM(cuDeviceGetName),
    CUDA_OVERRIDE_ENUM(cuDeviceCanAccessPeer),
    CUDA_OVERRIDE_ENUM(cuDeviceGetP2PAttribute),
    CUDA_OVERRIDE_ENUM(cuDeviceGetByPCIBusId),
    CUDA_OVERRIDE_ENUM(cuDeviceGetPCIBusId),
    CUDA_OVERRIDE_ENUM(cuDeviceGetUuid),
    CUDA_OVERRIDE_ENUM(cuDeviceGetDefaultMemPool),
    CUDA_OVERRIDE_ENUM(cuDeviceGetLuid),
    CUDA_OVERRIDE_ENUM(cuDeviceGetMemPool),
    CUDA_OVERRIDE_ENUM(cuDeviceTotalMem_v2),
    CUDA_OVERRIDE_ENUM(cuDriverGetVersion),
    CUDA_OVERRIDE_ENUM(cuDeviceGetTexture1DLinearMaxWidth),
    CUDA_OVERRIDE_ENUM(cuDeviceSetMemPool),
    CUDA_OVERRIDE_ENUM(cuFlushGPUDirectRDMAWrites),

    /* cuContext Part */
    CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxGetState),
    CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRetain),
    CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxSetFlags_v2),
    CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRelease_v2),
    CUDA_OVERRIDE_ENUM(cuCtxGetDevice),
    CUDA_OVERRIDE_ENUM(cuCtxCreate_v2),
    CUDA_OVERRIDE_ENUM(cuCtxDestroy_v2),
    CUDA_OVERRIDE_ENUM(cuCtxGetApiVersion),
    CUDA_OVERRIDE_ENUM(cuCtxGetCacheConfig),
    CUDA_OVERRIDE_ENUM(cuCtxGetCurrent),
    CUDA_OVERRIDE_ENUM(cuCtxGetFlags),
    CUDA_OVERRIDE_ENUM(cuCtxGetLimit),
    CUDA_OVERRIDE_ENUM(cuCtxGetSharedMemConfig),
    CUDA_OVERRIDE_ENUM(cuCtxGetStreamPriorityRange),
    CUDA_OVERRIDE_ENUM(cuCtxPopCurrent_v2),
    CUDA_OVERRIDE_ENUM(cuCtxPushCurrent_v2),
    CUDA_OVERRIDE_ENUM(cuCtxSetCacheConfig),
    CUDA_OVERRIDE_ENUM(cuCtxSetCurrent),
    CUDA_OVERRIDE_ENUM(cuCtxSetLimit),
    CUDA_OVERRIDE_ENUM(cuCtxSetSharedMemConfig),
    CUDA_OVERRIDE_ENUM(cuCtxSynchronize),
    //CUDA_OVERRIDE_ENUM(cuCtxEnablePeerAccess),
    CUDA_OVERRIDE_ENUM(cuGetExportTable),

    /* cuStream Part */
    CUDA_OVERRIDE_ENUM(cuStreamCreate),
    CUDA_OVERRIDE_ENUM(cuStreamDestroy_v2),
    CUDA_OVERRIDE_ENUM(cuStreamSynchronize),
    /* cuMemory Part */
    CUDA_OVERRIDE_ENUM(cuArray3DCreate_v2),
    CUDA_OVERRIDE_ENUM(cuArrayCreate_v2),
    CUDA_OVERRIDE_ENUM(cuArrayDestroy),
    CUDA_OVERRIDE_ENUM(cuMemAlloc_v2),
    CUDA_OVERRIDE_ENUM(cuMemAllocHost_v2),
    CUDA_OVERRIDE_ENUM(cuMemAllocManaged),
    CUDA_OVERRIDE_ENUM(cuMemAllocPitch_v2),
    CUDA_OVERRIDE_ENUM(cuMemFree_v2),
    CUDA_OVERRIDE_ENUM(cuMemFreeHost),
    CUDA_OVERRIDE_ENUM(cuMemHostAlloc),
    CUDA_OVERRIDE_ENUM(cuMemHostRegister_v2),
    CUDA_OVERRIDE_ENUM(cuMemHostUnregister),
    CUDA_OVERRIDE_ENUM(cuMemcpyDtoH_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpyHtoD_v2),
    CUDA_OVERRIDE_ENUM(cuMipmappedArrayCreate),
    CUDA_OVERRIDE_ENUM(cuMipmappedArrayDestroy),
    CUDA_OVERRIDE_ENUM(cuMemGetInfo_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpy),
    CUDA_OVERRIDE_ENUM(cuPointerGetAttribute),
    CUDA_OVERRIDE_ENUM(cuPointerGetAttributes),
    CUDA_OVERRIDE_ENUM(cuPointerSetAttribute),
    CUDA_OVERRIDE_ENUM(cuIpcCloseMemHandle),
    CUDA_OVERRIDE_ENUM(cuIpcGetMemHandle),
    CUDA_OVERRIDE_ENUM(cuIpcOpenMemHandle_v2),
    CUDA_OVERRIDE_ENUM(cuMemGetAddressRange_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpyAsync),
    CUDA_OVERRIDE_ENUM(cuMemcpyAtoD_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpyDtoA_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpyDtoD_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpyDtoDAsync_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpyDtoHAsync_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpyHtoDAsync_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpyPeer),
    CUDA_OVERRIDE_ENUM(cuMemcpyPeerAsync),
    CUDA_OVERRIDE_ENUM(cuMemsetD16_v2),
    CUDA_OVERRIDE_ENUM(cuMemsetD16Async),
    CUDA_OVERRIDE_ENUM(cuMemsetD2D16_v2),
    CUDA_OVERRIDE_ENUM(cuMemsetD2D16Async),
    CUDA_OVERRIDE_ENUM(cuMemsetD2D32_v2),
    CUDA_OVERRIDE_ENUM(cuMemsetD2D32Async),
    CUDA_OVERRIDE_ENUM(cuMemsetD2D8_v2),
    CUDA_OVERRIDE_ENUM(cuMemsetD2D8Async),
    CUDA_OVERRIDE_ENUM(cuMemsetD32_v2),
    CUDA_OVERRIDE_ENUM(cuMemsetD32Async),
    CUDA_OVERRIDE_ENUM(cuMemsetD8_v2),
    CUDA_OVERRIDE_ENUM(cuMemsetD8Async),
    CUDA_OVERRIDE_ENUM(cuMemAdvise),
    CUDA_OVERRIDE_ENUM(cuFuncSetCacheConfig),
    CUDA_OVERRIDE_ENUM(cuFuncSetSharedMemConfig),
    CUDA_OVERRIDE_ENUM(cuFuncGetAttribute),
    CUDA_OVERRIDE_ENUM(cuFuncSetAttribute),
    CUDA_OVERRIDE_ENUM(cuLaunchKernel),
    CUDA_OVERRIDE_ENUM(cuLaunchCooperativeKernel),
    /* cuEvent Part */
    CUDA_OVERRIDE_ENUM(cuEventCreate),
    CUDA_OVERRIDE_ENUM(cuEventDestroy_v2),
    CUDA_OVERRIDE_ENUM(cuModuleLoad),
    CUDA_OVERRIDE_ENUM(cuModuleLoadData),
    CUDA_OVERRIDE_ENUM(cuModuleLoadDataEx),
    CUDA_OVERRIDE_ENUM(cuModuleLoadFatBinary),
    CUDA_OVERRIDE_ENUM(cuModuleGetFunction),
    CUDA_OVERRIDE_ENUM(cuModuleUnload),
    CUDA_OVERRIDE_ENUM(cuModuleGetGlobal_v2),
    CUDA_OVERRIDE_ENUM(cuModuleGetTexRef),
    CUDA_OVERRIDE_ENUM(cuModuleGetSurfRef),
    CUDA_OVERRIDE_ENUM(cuLinkAddData_v2),
    CUDA_OVERRIDE_ENUM(cuLinkCreate_v2),
    CUDA_OVERRIDE_ENUM(cuLinkAddFile_v2),
    CUDA_OVERRIDE_ENUM(cuLinkComplete),
    CUDA_OVERRIDE_ENUM(cuLinkDestroy),

    /* Virtual Memory Part */
    CUDA_OVERRIDE_ENUM(cuMemAddressReserve),
    CUDA_OVERRIDE_ENUM(cuMemCreate),
    CUDA_OVERRIDE_ENUM(cuMemMap),
    CUDA_OVERRIDE_ENUM(cuMemAllocAsync),
    CUDA_OVERRIDE_ENUM(cuMemFreeAsync),
    /* cuda11.7 new api memory part */
    CUDA_OVERRIDE_ENUM(cuMemHostGetDevicePointer_v2),
    CUDA_OVERRIDE_ENUM(cuMemHostGetFlags),
    CUDA_OVERRIDE_ENUM(cuMemPoolTrimTo),
    CUDA_OVERRIDE_ENUM(cuMemPoolSetAttribute),
    CUDA_OVERRIDE_ENUM(cuMemPoolGetAttribute),
    CUDA_OVERRIDE_ENUM(cuMemPoolSetAccess),
    CUDA_OVERRIDE_ENUM(cuMemPoolGetAccess),
    CUDA_OVERRIDE_ENUM(cuMemPoolCreate),
    CUDA_OVERRIDE_ENUM(cuMemPoolDestroy),
    CUDA_OVERRIDE_ENUM(cuMemAllocFromPoolAsync),
    CUDA_OVERRIDE_ENUM(cuMemPoolExportToShareableHandle),
    CUDA_OVERRIDE_ENUM(cuMemPoolImportFromShareableHandle),
    CUDA_OVERRIDE_ENUM(cuMemPoolExportPointer),
    CUDA_OVERRIDE_ENUM(cuMemPoolImportPointer),
    CUDA_OVERRIDE_ENUM(cuMemcpy2DUnaligned_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpy2DAsync_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpy3D_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpy3DAsync_v2),
    CUDA_OVERRIDE_ENUM(cuMemcpy3DPeer),
    CUDA_OVERRIDE_ENUM(cuMemcpy3DPeerAsync),
    CUDA_OVERRIDE_ENUM(cuMemPrefetchAsync),
    CUDA_OVERRIDE_ENUM(cuMemRangeGetAttribute),
    CUDA_OVERRIDE_ENUM(cuMemRangeGetAttributes),
    /* cuda 11.7 external resource management */
    CUDA_OVERRIDE_ENUM(cuImportExternalMemory),
    CUDA_OVERRIDE_ENUM(cuExternalMemoryGetMappedBuffer),
    CUDA_OVERRIDE_ENUM(cuExternalMemoryGetMappedMipmappedArray),
    CUDA_OVERRIDE_ENUM(cuDestroyExternalMemory),
    CUDA_OVERRIDE_ENUM(cuImportExternalSemaphore),
    CUDA_OVERRIDE_ENUM(cuSignalExternalSemaphoresAsync),
    CUDA_OVERRIDE_ENUM(cuWaitExternalSemaphoresAsync),
    CUDA_OVERRIDE_ENUM(cuDestroyExternalSemaphore),
    /* cuda graph part */
    CUDA_OVERRIDE_ENUM(cuGraphCreate),
    CUDA_OVERRIDE_ENUM(cuGraphAddKernelNode_v2),
    CUDA_OVERRIDE_ENUM(cuGraphKernelNodeGetParams_v2),
    CUDA_OVERRIDE_ENUM(cuGraphKernelNodeSetParams_v2),
    CUDA_OVERRIDE_ENUM(cuGraphAddMemcpyNode),
    CUDA_OVERRIDE_ENUM(cuGraphMemcpyNodeGetParams),
    CUDA_OVERRIDE_ENUM(cuGraphMemcpyNodeSetParams),
    CUDA_OVERRIDE_ENUM(cuGraphAddMemsetNode),
    CUDA_OVERRIDE_ENUM(cuGraphMemsetNodeGetParams),
    CUDA_OVERRIDE_ENUM(cuGraphMemsetNodeSetParams),
    CUDA_OVERRIDE_ENUM(cuGraphAddHostNode),
    CUDA_OVERRIDE_ENUM(cuGraphHostNodeGetParams),
    CUDA_OVERRIDE_ENUM(cuGraphHostNodeSetParams),
    CUDA_OVERRIDE_ENUM(cuGraphAddChildGraphNode),
    CUDA_OVERRIDE_ENUM(cuGraphChildGraphNodeGetGraph),
    CUDA_OVERRIDE_ENUM(cuGraphAddEmptyNode),
    CUDA_OVERRIDE_ENUM(cuGraphAddEventRecordNode),
    CUDA_OVERRIDE_ENUM(cuGraphEventRecordNodeGetEvent),
    CUDA_OVERRIDE_ENUM(cuGraphEventRecordNodeSetEvent),
    CUDA_OVERRIDE_ENUM(cuGraphAddEventWaitNode),
    CUDA_OVERRIDE_ENUM(cuGraphEventWaitNodeGetEvent),
    CUDA_OVERRIDE_ENUM(cuGraphEventWaitNodeSetEvent),
    CUDA_OVERRIDE_ENUM(cuGraphAddExternalSemaphoresSignalNode),
    CUDA_OVERRIDE_ENUM(cuGraphExternalSemaphoresSignalNodeGetParams),
    CUDA_OVERRIDE_ENUM(cuGraphExternalSemaphoresSignalNodeSetParams),
    CUDA_OVERRIDE_ENUM(cuGraphAddExternalSemaphoresWaitNode),
    CUDA_OVERRIDE_ENUM(cuGraphExternalSemaphoresWaitNodeGetParams),
    CUDA_OVERRIDE_ENUM(cuGraphExternalSemaphoresWaitNodeSetParams),
    CUDA_OVERRIDE_ENUM(cuGraphExecExternalSemaphoresSignalNodeSetParams),
    CUDA_OVERRIDE_ENUM(cuGraphExecExternalSemaphoresWaitNodeSetParams),
    CUDA_OVERRIDE_ENUM(cuGraphClone),
    CUDA_OVERRIDE_ENUM(cuGraphNodeFindInClone),
    CUDA_OVERRIDE_ENUM(cuGraphNodeGetType),
    CUDA_OVERRIDE_ENUM(cuGraphGetNodes),
    CUDA_OVERRIDE_ENUM(cuGraphGetRootNodes),
    CUDA_OVERRIDE_ENUM(cuGraphGetEdges),
    CUDA_OVERRIDE_ENUM(cuGraphNodeGetDependencies),
    CUDA_OVERRIDE_ENUM(cuGraphNodeGetDependentNodes),
    CUDA_OVERRIDE_ENUM(cuGraphAddDependencies),
    CUDA_OVERRIDE_ENUM(cuGraphRemoveDependencies),
    CUDA_OVERRIDE_ENUM(cuGraphDestroyNode),
    CUDA_OVERRIDE_ENUM(cuGraphInstantiate),
    CUDA_OVERRIDE_ENUM(cuGraphInstantiateWithFlags),
    CUDA_OVERRIDE_ENUM(cuGraphUpload),
    CUDA_OVERRIDE_ENUM(cuGraphLaunch),
    CUDA_OVERRIDE_ENUM(cuGraphExecDestroy),
    CUDA_OVERRIDE_ENUM(cuGraphDestroy),

    CUDA_OVERRIDE_ENUM(cuGetProcAddress),
    CUDA_OVERRIDE_ENUM(cuGetProcAddress_v2),
    CUDA_ENTRY_END
}cuda_override_enum_t;

extern cuda_entry_t cuda_library_entry[];

#endif

#undef cuGetProcAddress
//CUresult hacked_cuGetProcAddress( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags );
#undef cuGraphInstantiate
CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);

CUresult hacked_cuInit(unsigned int Flags);
CUresult hacked_cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev );
CUresult hacked_cuDeviceGet(CUdevice *device,int ordinal);
CUresult hacked_cuDeviceGetCount( int* count );
CUresult hacked_cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult hacked_cuDeviceCanAccessPeer( int* canAccessPeer, CUdevice dev, CUdevice peerDev );
CUresult hacked_cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib,
                                 CUdevice srcDevice, CUdevice dstDevice);
CUresult hacked_cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId);
CUresult hacked_cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
CUresult hacked_cuDeviceGetUuid(CUuuid* uuid,CUdevice dev);
CUresult hacked_cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev);
CUresult hacked_cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev);
CUresult hacked_cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                         CUdevice dev);
CUresult hacked_cuDeviceTotalMem_v2 ( size_t* bytes, CUdevice dev );
CUresult hacked_cuDriverGetVersion(int *driverVersion);
CUresult hacked_cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev);
CUresult hacked_cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);
CUresult hacked_cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope);
CUresult hacked_cuDevicePrimaryCtxGetState( CUdevice dev, unsigned int* flags, int* active );
CUresult hacked_cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult hacked_cuDevicePrimaryCtxSetFlags_v2( CUdevice dev, unsigned int  flags );
CUresult hacked_cuDevicePrimaryCtxRelease_v2( CUdevice dev );
CUresult hacked_cuCtxGetDevice(CUdevice* device);
CUresult hacked_cuCtxCreate_v2 ( CUcontext* pctx, unsigned int  flags, CUdevice dev );
CUresult hacked_cuCtxDestroy_v2 ( CUcontext ctx );
CUresult hacked_cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version );
CUresult hacked_cuCtxGetCacheConfig ( CUfunc_cache* pconfig );
CUresult hacked_cuCtxGetCurrent ( CUcontext* pctx );
CUresult hacked_cuCtxGetFlags ( unsigned int* flags );
CUresult hacked_cuCtxGetLimit ( size_t* pvalue, CUlimit limit );
CUresult hacked_cuCtxGetSharedMemConfig ( CUsharedconfig* pConfig );
CUresult hacked_cuCtxGetStreamPriorityRange ( int* leastPriority, int* greatestPriority );
CUresult hacked_cuCtxPopCurrent_v2 ( CUcontext* pctx );
CUresult hacked_cuCtxPushCurrent_v2 ( CUcontext ctx );
CUresult hacked_cuCtxSetCacheConfig ( CUfunc_cache config );
CUresult hacked_cuCtxSetCurrent ( CUcontext ctx );
CUresult hacked_cuCtxSetLimit ( CUlimit limit, size_t value );
CUresult hacked_cuCtxSetSharedMemConfig ( CUsharedconfig config );
CUresult hacked_cuCtxSynchronize ( void );
CUresult hacked_cuGetProcAddress ( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags );
CUresult hacked_cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);

CUresult hacked_cuArray3DCreate_v2(CUarray* arr, const CUDA_ARRAY3D_DESCRIPTOR* desc);
CUresult hacked_cuArrayCreate_v2(CUarray* arr, const CUDA_ARRAY_DESCRIPTOR* desc);
CUresult hacked_cuArrayDestroy(CUarray arr);
CUresult hacked_cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
CUresult hacked_cuMemAllocHost_v2(void** hptr, size_t bytesize);
CUresult hacked_cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
CUresult hacked_cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, 
                                      size_t Height, unsigned int ElementSizeBytes);
CUresult hacked_cuMemFree_v2(CUdeviceptr dptr);
CUresult hacked_cuMemFreeHost(void* hptr);
CUresult hacked_cuMemHostAlloc(void** hptr, size_t bytesize, unsigned int flags);
CUresult hacked_cuMemHostRegister_v2(void* hptr, size_t bytesize, unsigned int flags);
CUresult hacked_cuMemHostUnregister(void* hptr);
CUresult hacked_cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount );
CUresult hacked_cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr );
CUresult hacked_cuPointerGetAttributes ( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr );
CUresult hacked_cuPointerSetAttribute ( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr );
CUresult hacked_cuIpcCloseMemHandle(CUdeviceptr dptr);
CUresult hacked_cuIpcGetMemHandle ( CUipcMemHandle* pHandle, CUdeviceptr dptr );
CUresult hacked_cuIpcOpenMemHandle_v2 ( CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags );
CUresult hacked_cuMemGetAddressRange_v2( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr );
CUresult hacked_cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream );
CUresult hacked_cuMemcpyAtoD_v2( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount );
CUresult hacked_cuMemcpyDtoA_v2 ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount );
CUresult hacked_cuMemcpyDtoD_v2 ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount );
CUresult hacked_cuMemcpyDtoDAsync_v2( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream );
CUresult hacked_cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult hacked_cuMemcpyDtoHAsync_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult hacked_cuMemcpyHtoD_v2(CUdeviceptr srcDevice, const void* dstHost, size_t ByteCount);
CUresult hacked_cuMemcpyHtoDAsync_v2( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream );
CUresult hacked_cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
CUresult hacked_cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
CUresult hacked_cuMemsetD16_v2 ( CUdeviceptr dstDevice, unsigned short us, size_t N );
CUresult hacked_cuMemsetD16Async ( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream );
CUresult hacked_cuMemsetD2D16_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height );
CUresult hacked_cuMemsetD2D16Async (CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream );
CUresult hacked_cuMemsetD2D32_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height );
CUresult hacked_cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream );
CUresult hacked_cuMemsetD2D8_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height );
CUresult hacked_cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream );
CUresult hacked_cuMemsetD32_v2 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N );
CUresult hacked_cuMemsetD32Async ( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream );
CUresult hacked_cuMemsetD8_v2 ( CUdeviceptr dstDevice, unsigned char  uc, size_t N );
CUresult hacked_cuMemsetD8Async ( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream );
CUresult hacked_cuMemAdvise( CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device );
CUresult hacked_cuMemGetInfo_v2(size_t* free, size_t* total);
CUresult hacked_cuMipmappedArrayCreate(CUmipmappedArray* pHandle, 
                                          const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, 
                                          unsigned int numMipmapLevels);
CUresult hacked_cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
CUresult hacked_cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra );
CUresult hacked_cuLaunchCooperativeKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams );
CUresult hacked_cuMemAddressReserve ( CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags );
CUresult hacked_cuMemCreate ( CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags );
CUresult hacked_cuMemMap( CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags );
CUresult hacked_cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
CUresult hacked_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult hacked_cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags);
CUresult hacked_cuMemHostGetFlags(unsigned int *pFlags, void *p);
CUresult hacked_cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
CUresult hacked_cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
CUresult hacked_cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value); 
CUresult hacked_cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count);
CUresult hacked_cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location);
CUresult hacked_cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps);
CUresult hacked_cuMemPoolDestroy(CUmemoryPool pool);
CUresult hacked_cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
CUresult hacked_cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags);
CUresult hacked_cuMemPoolImportFromShareableHandle(
        CUmemoryPool *pool_out,
        void *handle,
        CUmemAllocationHandleType handleType,
        unsigned long long flags);
CUresult hacked_cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr);
CUresult hacked_cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData);
CUresult hacked_cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy);
CUresult hacked_cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
CUresult hacked_cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy);
CUresult hacked_cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
CUresult hacked_cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy);
CUresult hacked_cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream);
CUresult hacked_cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
CUresult hacked_cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);
CUresult hacked_cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);
CUresult hacked_cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc);
CUresult hacked_cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc);
CUresult hacked_cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc);
CUresult hacked_cuDestroyExternalMemory(CUexternalMemory extMem);
CUresult hacked_cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc);
CUresult hacked_cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
CUresult hacked_cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
CUresult hacked_cuDestroyExternalSemaphore(CUexternalSemaphore extSem);
CUresult hacked_cuEventCreate ( CUevent* phEvent, unsigned int  Flags );
CUresult hacked_cuEventDestroy_v2 ( CUevent hEvent );
CUresult hacked_cuModuleLoad ( CUmodule* module, const char* fname );
CUresult hacked_cuModuleLoadData( CUmodule* module, const void* image);
CUresult hacked_cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues );
CUresult hacked_cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin );
CUresult hacked_cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name );
CUresult hacked_cuModuleUnload(CUmodule hmod);
CUresult hacked_cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult hacked_cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
CUresult hacked_cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name);
CUresult hacked_cuLinkAddData_v2 ( CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues );
CUresult hacked_cuLinkCreate_v2 ( unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut );
CUresult hacked_cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path,
    unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult hacked_cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut);
CUresult hacked_cuLinkDestroy(CUlinkState state);
CUresult hacked_cuFuncSetCacheConfig ( CUfunction hfunc, CUfunc_cache config );
CUresult hacked_cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);
CUresult hacked_cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult hacked_cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult hacked_cuStreamCreate(CUstream *phstream, unsigned int flags);
CUresult hacked_cuStreamDestroy_v2 ( CUstream hStream);
CUresult hacked_cuStreamSynchronize(CUstream hstream);
CUresult hacked_cuGraphCreate(CUgraph *phGraph, unsigned int flags);
CUresult hacked_cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
CUresult hacked_cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams);
CUresult hacked_cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams);
CUresult hacked_cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
CUresult hacked_cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph);
CUresult hacked_cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph);
CUresult hacked_cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies);
CUresult hacked_cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
CUresult hacked_cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
CUresult hacked_cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event);
CUresult hacked_cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
CUresult hacked_cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
CUresult hacked_cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event);
CUresult hacked_cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out);
CUresult hacked_cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out);
CUresult hacked_cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
CUresult hacked_cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph);
CUresult hacked_cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
CUresult hacked_cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type);
CUresult hacked_cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes);
CUresult hacked_cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes);
CUresult hacked_cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges);
CUresult hacked_cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies);
CUresult hacked_cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes);
CUresult hacked_cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
CUresult hacked_cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
CUresult hacked_cuGraphDestroyNode(CUgraphNode hNode);
CUresult hacked_cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
CUresult hacked_cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags);
CUresult hacked_cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream);
CUresult hacked_cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
CUresult hacked_cuGraphExecDestroy(CUgraphExec hGraphExec);
CUresult hacked_cuGraphDestroy(CUgraph hGraph);

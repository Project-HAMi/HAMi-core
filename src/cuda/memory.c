#include <dirent.h>
#include <time.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "include/memory_limit.h"

extern int pidfound;

extern void rate_limiter(int64_t grids, int64_t blocks);

int check_oom() {
//    return 0;
    CUdevice dev;
    CHECK_DRV_API(cuCtxGetDevice(&dev));
    return oom_check(dev,0);
}

CUresult cuArray3DCreate_v2(CUarray* arr, const CUDA_ARRAY3D_DESCRIPTOR* desc) {
    LOG_DEBUG("cuArray3DCreate_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuArray3DCreate_v2, arr, desc);
}


CUresult cuArrayCreate_v2(CUarray* arr, const CUDA_ARRAY_DESCRIPTOR* desc) {
    LOG_DEBUG("cuArrayCreate_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuArrayCreate_v2, arr, desc);
}


CUresult cuArrayDestroy(CUarray arr) {
    LOG_DEBUG("cuArrayDestroy");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuArrayDestroy, arr);
}

CUresult cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, void* data) {
    CUresult res;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAlloc_v2,dptr,bytesize);
    return res;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    MEMORY_OP_SCOPE;
    LOG_INFO("into cuMemAllocing_v2 dptr=%p bytesize=%ld",dptr,bytesize);
    ENSURE_RUNNING();
    CUresult res = allocate_raw(dptr,bytesize);
    if (res!=CUDA_SUCCESS)
        return res;
    LOG_INFO("res=%d, cuMemAlloc_v2 success dptr=%p bytesize=%lu",0,(void *)*dptr,bytesize);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocHost_v2(void** hptr, size_t bytesize) {
    LOG_DEBUG("cuMemAllocHost_v2 hptr=%p bytesize=%ld",hptr,bytesize);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocHost_v2, hptr, bytesize);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    if (check_oom()) {
        CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeHost, *hptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return res;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    MEMORY_OP_SCOPE;
    LOG_DEBUG("cuMemAllocManaged dptr=%p bytesize=%ld",dptr,bytesize);
    ENSURE_RUNNING();
    CUdevice dev;
    CHECK_DRV_API(cuCtxGetDevice(&dev));
    if (reserve_device_memory(dev, bytesize) != 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocManaged, dptr, bytesize, flags);
    if (res == CUDA_SUCCESS) {
        if (add_chunk_only(*dptr, bytesize, dev) != 0) {
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemFree_v2, *dptr);
            release_device_memory(dev, bytesize);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    } else {
        release_device_memory(dev, bytesize);
    }
    return res;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, 
                                      size_t Height, unsigned int ElementSizeBytes) {
    MEMORY_OP_SCOPE;
    LOG_DEBUG("cuMemAllocPitch_v2 dptr=%p (%ld,%ld)",dptr,WidthInBytes,Height);
    size_t guess_pitch = (ElementSizeBytes == 0 || WidthInBytes == 0) ? 0 :
        (((WidthInBytes - 1) / ElementSizeBytes) + 1) * ElementSizeBytes;
    size_t bytesize = guess_pitch * Height;
    size_t actual;
    ENSURE_RUNNING();
    CUdevice dev;
    CHECK_DRV_API(cuCtxGetDevice(&dev));
    if (reserve_device_memory(dev, bytesize) != 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocPitch_v2, dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    if (res != CUDA_SUCCESS) {
        release_device_memory(dev, bytesize);
        return res;
    }
    /* Driver pitch may exceed guess_pitch due to alignment; account for real size. */
    if (Height != 0 && pPitch != NULL && *pPitch > SIZE_MAX / Height) {
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemFree_v2, *dptr);
        release_device_memory(dev, bytesize);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    actual = (pPitch != NULL) ? (*pPitch * Height) : bytesize;
    if (actual > bytesize) {
        if (reserve_device_memory(dev, actual - bytesize) != 0) {
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemFree_v2, *dptr);
            release_device_memory(dev, bytesize);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    } else if (actual < bytesize) {
        release_device_memory(dev, bytesize - actual);
    }
    if (add_chunk_only(*dptr, actual, dev) != 0) {
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemFree_v2, *dptr);
        release_device_memory(dev, actual);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    MEMORY_OP_SCOPE;
    LOG_DEBUG("cuMemFree_v2 dptr=%llx",dptr);
    if (dptr == 0) {  // NULL
        return CUDA_SUCCESS;
    }
    CUresult res = free_raw(dptr);
    if (res == CUDA_SUCCESS) graph_ptr_freed(dptr);
    LOG_INFO("after free_raw dptr=%p res=%d",(void *)dptr,res);
    return res;
}


CUresult cuMemFreeHost(void* hptr) {
    LOG_DEBUG("cuMemFreeHost_v2 hptr=%p",hptr);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeHost, hptr);
    return res;
}

CUresult cuMemHostAlloc(void** hptr, size_t bytesize, unsigned int flags) {
    LOG_DEBUG("cuMemHostAlloc hptr=%p bytesize=%lu",hptr,bytesize);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostAlloc, hptr, bytesize, flags);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    if (check_oom()) {
        CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeHost, *hptr);
        *hptr = NULL;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return res;
}


CUresult cuMemHostRegister_v2(void* hptr, size_t bytesize, unsigned int flags) {
    // TODO: process flags properly
    LOG_DEBUG("cuMemHostRegister_v2 hptr=%p bytesize=%ld",hptr,bytesize);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostRegister_v2, hptr, bytesize, flags);
    LOG_DEBUG("cuMemHostRegister_v2 returned :%d(%p:%ld)",res,hptr,bytesize);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    if (check_oom()) {
        CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostUnregister, hptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return res;
}


CUresult cuMemHostUnregister(void* hptr) {
    LOG_DEBUG("cuMemHostUnregister hptr=%p",hptr);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostUnregister, hptr);
    return res;
}


CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount ){
    LOG_DEBUG("cuMemcpy dst=%llx src=%llx count=%lu",dst,src,ByteCount);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy,dst,src,ByteCount);
    return res;
}

CUresult cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr ){
    LOG_DEBUG("cuPointGetAttribute data=%p attribute=%d ptr=%llx", data, (int)attribute,ptr);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuPointerGetAttribute,data,attribute,ptr);
    return res;
}

CUresult cuPointerGetAttributes ( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr ) {
    LOG_DEBUG("cuPointGetAttribute data=%p ptr=%llx", data, ptr);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuPointerGetAttributes,numAttributes,attributes,data,ptr);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    int cur=0;
    for (cur=0;cur<numAttributes;cur++){
        if (attributes[cur]==CU_POINTER_ATTRIBUTE_MEMORY_TYPE){
            int j = check_memory_type(ptr);
            LOG_DEBUG("check result = %d %d",j,*(int *)(data[cur]));
        }
    }
    return res;
}

CUresult cuPointerSetAttribute ( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr ){
    LOG_DEBUG("cuPointSetAttribute value=%p attribute=%d ptr=%llx", value, (int)attribute, ptr);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuPointerSetAttribute,value,attribute,ptr);
    return res;
}


CUresult cuIpcCloseMemHandle(CUdeviceptr dptr){
    LOG_DEBUG("cuIpcCloseMemHandle dptr=%llx",dptr);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuIpcCloseMemHandle,dptr);
}

CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
    LOG_MSG("cuIpcGetMemHandle dptr=%llx", dptr);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuIpcGetMemHandle,pHandle,dptr);
}

CUresult cuIpcOpenMemHandle_v2 ( CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags ){
    LOG_DEBUG("cuIpcGetMemHandle dptr=%p",pdptr);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuIpcOpenMemHandle_v2,pdptr,handle,Flags);
}


CUresult cuMemGetAddressRange_v2( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr ){
    //TODO: Translate back
    LOG_DEBUG("cuMemGetAddressRange_v2,dptr=%llx",dptr);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemGetAddressRange_v2,pbase,psize,dptr);
    return res;
}

CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream ){
    LOG_DEBUG("cuMemcpyAsync,dst=%llx src=%llx count=%lu",dst,src,ByteCount);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyAsync,dst,src,ByteCount,hStream);
    return res; 
}

CUresult cuMemcpyAtoD_v2( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount ){
    LOG_DEBUG("cuMemcpyAtoD_v2,dst=%llx count=%lu",dstDevice,ByteCount);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyAtoD_v2,dstDevice,srcArray,srcOffset,ByteCount);
}

CUresult cuMemcpyDtoA_v2 ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount ){
    LOG_DEBUG("cuMemcpyDtoA_v2,src=%llx count=%lu",srcDevice,ByteCount);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoA_v2,dstArray,dstOffset,srcDevice,ByteCount);
}

CUresult cuMemcpyDtoD_v2 ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount ){
    LOG_DEBUG("cuMemcpyDtoD_v2,dst=%llx src=%llx count=%lu",dstDevice,srcDevice,ByteCount);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoD_v2,dstDevice,srcDevice,ByteCount);
}

CUresult cuMemcpyDtoDAsync_v2( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ){
    LOG_DEBUG("cuMemcpyDtoDAsync_v2,dst=%llx src=%llx count=%lu",dstDevice,srcDevice,ByteCount);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoDAsync_v2,dstDevice,srcDevice,ByteCount,hStream);
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    // TODO: compute bytesize
    LOG_DEBUG("cuMemcpyDtoH_v2,dst=%p src=%llx count=%lu",dstHost,srcDevice,ByteCount);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoH_v2, dstHost, srcDevice, ByteCount);
    return res;
}

CUresult cuMemcpyDtoHAsync_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ){
    LOG_DEBUG("cuMemcpyDtoHAsync_v2,dst=%p src=%llx count=%lu",dstHost,srcDevice,ByteCount);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoHAsync_v2,dstHost,srcDevice,ByteCount,hStream); 
}


CUresult cuMemcpyHtoD_v2(CUdeviceptr srcDevice, const void* dstHost, size_t ByteCount) {
    // TODO: compute bytesize
    LOG_DEBUG("cuMemcpyHtoD_v2,srcDevice=%llx dstHost=%p count=%lu",srcDevice,dstHost,ByteCount);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyHtoD_v2, srcDevice, dstHost, ByteCount);
    return res;
}

CUresult cuMemcpyHtoDAsync_v2( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream ){
    LOG_DEBUG("cuMemcpyHtoDAsync_v2,dst=%llx src=%p count=%lu",dstDevice,srcHost,ByteCount);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyHtoDAsync_v2,dstDevice,srcHost,ByteCount,hStream);
    return res;
}


CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
    LOG_DEBUG("cuMemcpyPeer dstDevice=%llx srcDevice=%llx",dstDevice,srcDevice);
    ENSURE_RUNNING();
    CUresult res=CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyPeer,dstDevice,dstContext,srcDevice,srcContext,ByteCount);
    return res;
}

CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream){
    LOG_DEBUG("into cuMemcpyPeerAsync dstDevice=%llx srcDevice=%llx",dstDevice,srcDevice);
    ENSURE_RUNNING();
    CUresult res=CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyPeerAsync,dstDevice,dstContext,srcDevice,srcContext,ByteCount,hStream);
    return res;
}

CUresult cuMemsetD16_v2 ( CUdeviceptr dstDevice, unsigned short us, size_t N ){
    ENSURE_RUNNING();
    CUresult res=CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD16_v2,dstDevice,us,N);
    return res;
}

CUresult cuMemsetD16Async ( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD16Async,dstDevice,us,N,hStream);
}

CUresult cuMemsetD2D16_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D16_v2,dstDevice,dstPitch,us,Width,Height);
}

CUresult cuMemsetD2D16Async (CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D16Async,dstDevice,dstPitch,us,Width,Height,hStream);
}

CUresult cuMemsetD2D32_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D32_v2,dstDevice,dstPitch,ui,Width,Height);
}


CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D32Async,dstDevice,dstPitch,ui,Width,Height,hStream);
}

CUresult cuMemsetD2D8_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D8_v2,dstDevice,dstPitch,uc,Width,Height);
}

CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D8Async,dstDevice,dstPitch,uc,Width,Height,hStream);
}

CUresult cuMemsetD32_v2 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N ){
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD32_v2,dstDevice,ui,N);
    return res;
}

CUresult cuMemsetD32Async ( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD32Async,dstDevice,ui,N,hStream);
}   


CUresult cuMemsetD8_v2 ( CUdeviceptr dstDevice, unsigned char  uc, size_t N ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD8_v2,dstDevice,uc,N);
}

CUresult cuMemsetD8Async ( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD8Async,dstDevice,uc,N,hStream);
}

#if CUDA_VERSION < 13000
CUresult cuMemAdvise( CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device ){
    LOG_DEBUG("cuMemAdvise devPtr=%llx count=%lx",devPtr,count);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAdvise,devPtr,count,advice,device);
}
#endif

/* On CUDA 13+, cuda.h #defines cuMemAdvise as cuMemAdvise_v2 with a new CUmemLocation
   parameter. The old CUdevice wrapper above would conflict, so it is excluded.
   On CUDA 12 and below, both symbols exist separately and this is the explicit v2 wrapper. */
CUresult cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location) {
    LOG_DEBUG("cuMemAdvise_v2 devPtr=%llx count=%lx", devPtr, count);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemAdvise_v2, devPtr, count, advice, location);
}

#ifdef HOOK_MEMINFO_ENABLE
CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
    CUdevice dev;
    size_t drv_free = 0, drv_total = 0, out_free, out_total;
    LOG_DEBUG("cuMemGetInfo_v2");
    ENSURE_INITIALIZED();
    ensure_hostpid();
    CHECK_DRV_API(cuCtxGetDevice(&dev));
    size_t usage = get_current_device_memory_usage(cuda_to_nvml_map(dev));
    size_t limit = get_current_device_memory_limit(dev);
    /* libnvoptix calls this with free=NULL, which the driver accepts, so read
       into locals and write back only what the caller asked for. */
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemGetInfo_v2,
                                      &drv_free, &drv_total);
    if (res != CUDA_SUCCESS) {
        return res;
    }

    if (limit == 0) {
        out_total = drv_total;
        out_free = drv_total - usage;
    } else {
        // Ensure total memory does not exceed the physical or imposed limit.
        size_t actual_limit = (limit > drv_total) ? drv_total : limit;
        size_t clamped = (usage > limit) ? limit : usage;
        if (usage > limit) {
            LOG_WARN("CUDA meminfo: usage %lu exceeds limit %lu, clamping", usage, limit);
        }
        out_total = actual_limit;
        out_free = (actual_limit > clamped) ? (actual_limit - clamped) : 0;
    }
    LOG_INFO("cuMemGetInfo_v2 drv_free=%ld drv_total=%ld free=%ld total=%ld limit=%ld usage=%ld",
        drv_free, drv_total, out_free, out_total, limit, usage);

    if (free != NULL) {
        *free = out_free;
    }
    if (total != NULL) {
        *total = out_total;
    }
    return CUDA_SUCCESS;
}
#endif

CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, 
                                          const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, 
                                          unsigned int numMipmapLevels) {
    // TODO: compute bytesize
    LOG_DEBUG("cuMipmappedArrayCreate\n");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMipmappedArrayCreate, pHandle, pMipmappedArrayDesc, numMipmapLevels);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    if (check_oom()) {
        CUDA_OVERRIDE_CALL(cuda_library_entry,cuMipmappedArrayDestroy, *pHandle);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return res;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    // TODO: compute bytesize
    LOG_DEBUG("cuMipmappedArrayDestroy\n");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMipmappedArrayDestroy, hMipmappedArray);
    return res;
}

CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ){
    ENSURE_RUNNING();
    pre_launch_kernel();
    if (pidfound==1){ 
        rate_limiter((int64_t)gridDimX * gridDimY * gridDimZ,
                   (int64_t)blockDimX * blockDimY * blockDimZ);
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuLaunchKernel,f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra);
    return res;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra) {
    ENSURE_RUNNING();
    pre_launch_kernel();
    if (pidfound==1){
        rate_limiter((int64_t)config->gridDimX * config->gridDimY * config->gridDimZ,
                   (int64_t)config->blockDimX * config->blockDimY * config->blockDimZ);
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuLaunchKernelEx,config,f,kernelParams,extra);
    return res;
}

CUresult cuLaunchCooperativeKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams ){
    ENSURE_RUNNING();
    pre_launch_kernel();
    if (pidfound==1){
        rate_limiter((int64_t)gridDimX * gridDimY * gridDimZ,
                   (int64_t)blockDimX * blockDimY * blockDimZ);
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuLaunchCooperativeKernel,f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams);
    return res;
}

CUresult cuMemoryFree(CUdeviceptr dptr) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFree_v2,dptr);
    return res;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size,
    size_t alignment, CUdeviceptr addr, unsigned long long flags ) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
        cuMemAddressReserve, ptr, size, alignment, addr, flags);
    LOG_INFO("cuMemAddressReserve:%lx %llx", size, *ptr);
    return res;
}

CUresult cuMemCreate ( CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags ) {
    MEMORY_OP_SCOPE;
    LOG_INFO("cuMemCreate:%lld:%d", size, prop->location.id);
    ENSURE_RUNNING();
    CUdevice dev;
    int do_oom_check = (prop->location.type == CU_MEM_LOCATION_TYPE_DEVICE);
    if (do_oom_check && cuCtxGetDevice(&dev) != CUDA_SUCCESS) {
        dev = prop->location.id;
    }
    if (do_oom_check && reserve_device_memory(dev, size) != 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
        cuMemCreate, handle, size, prop, flags);
    if (do_oom_check) {
        if (res == CUDA_SUCCESS) {
            if (add_chunk_only(*handle, size, dev) != 0) {
                CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemRelease, *handle);
                release_device_memory(dev, size);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            if (vmm_handle_created(*handle) != 0) {
                remove_chunk_only(*handle);
                CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemRelease, *handle);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        } else {
            release_device_memory(dev, size);
        }
    }
    return res;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    MEMORY_OP_SCOPE;
    LOG_INFO("cuMemRelease:%llx", handle);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemRelease, handle);
    if (res == CUDA_SUCCESS) {
        vmm_handle_released(handle);
    }
    return res;
}

CUresult cuMemMap( CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags ) {
    MEMORY_OP_SCOPE;
    LOG_INFO("cuMemMap:%lld(%llx,%llx)", size, ptr, offset);
    vmm_map_entry *m = NULL;
    if (vmm_map_begin(handle, &m) != 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemMap,ptr,size,offset,handle,flags);
    vmm_map_end(m, ptr, res == CUDA_SUCCESS);
    return res;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    MEMORY_OP_SCOPE;
    LOG_INFO("cuMemUnmap:%lld(%llx)", size, ptr);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemUnmap, ptr, size);
    if (res == CUDA_SUCCESS) {
        vmm_unmapped(ptr, size);
    }
    return res;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
    LOG_INFO("cuMemRetainAllocationHandle:%p", addr);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemRetainAllocationHandle, handle, addr);
    if (res == CUDA_SUCCESS) {
        vmm_handle_retained(*handle);
    }
    return res;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
    void* osHandle, CUmemAllocationHandleType shHandleType) {
    LOG_INFO("cuMemImportFromSharableHandle");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
        cuMemImportFromShareableHandle, handle, osHandle, shHandleType);
    return res;
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    MEMORY_OP_SCOPE;
    LOG_DEBUG("cuMemAllocAsync:%ld",bytesize);
    return allocate_async_raw(dptr,bytesize,hStream);
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    MEMORY_OP_SCOPE;
    LOG_DEBUG("cuMemFreeAsync dptr=%llx",dptr);
    if (dptr == 0) {  // NULL
        return CUDA_SUCCESS;
    }
    CUresult res = free_raw_async(dptr,hStream);
    LOG_DEBUG("after free_raw_async dptr=%p res=%d",(void *)dptr,res);
    return res;
}

/* cuGraphClone remains untracked. */
/** Reserve graph bookkeeping and quota before recording an allocation node. */
CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph,
        const CUgraphNode *dependencies, size_t numDependencies,
        CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
    LOG_DEBUG("cuGraphAddMemAllocNode");
    int charge = nodeParams != NULL &&
        nodeParams->poolProps.location.type == CU_MEM_LOCATION_TYPE_DEVICE;
    CUdevice dev = charge ? nodeParams->poolProps.location.id : 0;
    size_t size = charge ? nodeParams->bytesize : 0;
    graph_charge_entry *entry = NULL;
    if (charge) {
        cuuint64_t id = graph_capture_id_of(hGraph);
        CUresult res = graph_alloc_begin(hGraph, dev, size, id, id != 0, &entry);
        if (res != CUDA_SUCCESS) return res;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphAddMemAllocNode,
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    graph_alloc_end(entry, res == CUDA_SUCCESS && charge ? nodeParams->dptr : 0, res);
    return res;
}


/** Reserve allocation demand and preserve native generic-node parameters and edge data. */
static CUresult graph_add_node(cuda_sym_t add, int edges, CUgraphNode *node, CUgraph graph,
                                const CUgraphNode *deps, const CUgraphEdgeData *data,
                                size_t count, CUgraphNodeParams *params) {
    if (add == NULL) return CUDA_ERROR_NOT_SUPPORTED;
#if CUDA_VERSION >= 13000
    CUgraph child = params != NULL && params->type == CU_GRAPH_NODE_TYPE_GRAPH &&
        params->graph.ownership == CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE ? params->graph.graph : NULL;
#endif
    int charge = params != NULL && params->type == CU_GRAPH_NODE_TYPE_MEM_ALLOC &&
        params->alloc.poolProps.location.type == CU_MEM_LOCATION_TYPE_DEVICE;
    graph_charge_entry *entry = NULL;
    if (charge) {
        cuuint64_t id = graph_capture_id_of(graph);
        CUresult res = graph_alloc_begin(graph, params->alloc.poolProps.location.id,
            params->alloc.bytesize, id, id != 0, &entry);
        if (res != CUDA_SUCCESS) return res;
    }
#if CUDA_VERSION >= 13000
    if (child != NULL) {
        CUresult res = graph_child_begin(graph, child);
        if (res != CUDA_SUCCESS) return res;
    }
#endif
    CUresult res = edges ? add(node, graph, deps, data, count, params)
                         : add(node, graph, deps, count, params);
#if CUDA_VERSION >= 13000
    if (child != NULL) graph_child_end(graph, child, res);
#endif
    graph_alloc_end(entry, res == CUDA_SUCCESS && charge ? params->alloc.dptr : 0, res);
    if (res == CUDA_SUCCESS && params != NULL && params->type == CU_GRAPH_NODE_TYPE_MEM_FREE)
        graph_mark_free(graph, params->free.dptr);
    return res;
}

CUresult cuGraphAddNode(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                        size_t count, CUgraphNodeParams *params) {
    return graph_add_node((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry, cuGraphAddNode),
        0, node, graph, deps, NULL, count, params);
}

CUresult cuGraphAddNode_v2(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                           const CUgraphEdgeData *data, size_t count, CUgraphNodeParams *params) {
    return graph_add_node((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry, cuGraphAddNode_v2),
        1, node, graph, deps, data, count, params);
}

CUresult cuGraphAddNode_v3(CUgraphNode *node, CUgraph graph, const CUgraphNode *deps,
                           const CUgraphEdgeData *data, size_t count, CUgraphNodeParams *params) {
    return graph_add_node((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry, cuGraphAddNode_v3),
        1, node, graph, deps, data, count, params);
}

CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph,
        const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr) {
    LOG_DEBUG("cuGraphAddMemFreeNode");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphAddMemFreeNode,
        phGraphNode, hGraph, dependencies, numDependencies, dptr);
    if (res == CUDA_SUCCESS) {
        graph_mark_free(hGraph, dptr);
    }
    return res;
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags){
    LOG_INFO("cuMemHostGetDevicePointer");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostGetDevicePointer_v2,pdptr,p,Flags);
}

CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p){
    LOG_INFO("cuMemHostGetFlags");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostGetFlags,pFlags,p);
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep){
    MEMORY_OP_SCOPE;
    LOG_DEBUG("cuMemPoolTrimTo");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolTrimTo,pool,minBytesToKeep);
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    LOG_DEBUG("cuMemPoolSetAttribute");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolSetAttribute,pool,attr,value);
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAttribute,pool,attr,value);
    LOG_INFO("cuMemPoolGetAttribute %d %ld",attr,*(long *)value);
    return res;
}

CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
    LOG_DEBUG("cuMemPoolSetAccess");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolSetAccess,pool,map,count);
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location) {
    LOG_DEBUG("cuMemPoolGetAccess");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAccess,flags,memPool,location);
}

CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
    LOG_DEBUG("cuMemPoolCreate");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolCreate,pool,poolProps);
}

CUresult cuMemPoolDestroy(CUmemoryPool pool) {
    LOG_DEBUG("cuMemPoolDestroy");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolDestroy,pool);
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    MEMORY_OP_SCOPE;
    LOG_DEBUG("cuMemAllocFromPoolAsync:%ld", bytesize);
    return allocate_from_pool_async_raw(dptr, bytesize, pool, hStream);
}

CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) {
    LOG_DEBUG("cuMemPoolExportToShareableHandle");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolExportToShareableHandle,handle_out,pool,handleType,flags);
}

CUresult cuMemPoolImportFromShareableHandle(
        CUmemoryPool *pool_out,
        void *handle,
        CUmemAllocationHandleType handleType,
        unsigned long long flags) {
            LOG_DEBUG("cuMemPoolImportFromShareableHandle");
            return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolImportFromShareableHandle,pool_out,handle,handleType,flags);
        }

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr) {
    LOG_DEBUG("cuMemPoolExportPointer");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolExportPointer,shareData_out,ptr);
}

CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData) {
    LOG_DEBUG("cuMemPoolImportPointer");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolImportPointer,ptr_out,pool,shareData);
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
    LOG_DEBUG("cuMemcpy2D_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpy2D, pCopy);
}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
    LOG_DEBUG("cuMemcpy2DUnaligned_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy2DUnaligned_v2,pCopy);
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    LOG_DEBUG("cuMemcpy2DAsync_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy2DAsync,pCopy,hStream);
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
    LOG_DEBUG("cuMemcpy3D_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy3D_v2,pCopy);
}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    LOG_DEBUG("cuMemcpy3DAsync_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy3DAsync_v2,pCopy,hStream);
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
    LOG_DEBUG("cuMemcpy3DPeer");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy3DPeer,pCopy);
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
    LOG_DEBUG("cuMemcpy3DPeerAsync");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy3DPeerAsync,pCopy,hStream);
}

#if CUDA_VERSION < 13000
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
    LOG_DEBUG("cuMemPrefetchAsync");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPrefetchAsync,devPtr,count,dstDevice,hStream);
}
#endif

CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags,
                               CUstream hStream) {
    LOG_DEBUG("cuMemPrefetchAsync_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemPrefetchAsync_v2, devPtr, count, location, flags, hStream);
}

CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
    LOG_DEBUG("cuMemRangeGetAttribute");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemRangeGetAttribute,data,dataSize,attribute,devPtr,count);
}

CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) {
    LOG_DEBUG("cuMemRangeGetAttributes");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemRangeGetAttributes,data,dataSizes,attributes,numAttributes,devPtr,count);
}

/* External Resource Management */
CUresult cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
    LOG_DEBUG("cuImportExternalMemory");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuImportExternalMemory,extMem_out,memHandleDesc);
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
    LOG_DEBUG("cuExternalMemoryGetMappedBuffer");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuExternalMemoryGetMappedBuffer,devPtr,extMem,bufferDesc);
}

CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
    LOG_DEBUG("cuExternalMemoryGetMappedMipmappedArray");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuExternalMemoryGetMappedMipmappedArray,mipmap,extMem,mipmapDesc);
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
    LOG_DEBUG("cuDestroyExternalMemory");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDestroyExternalMemory,extMem);
}

CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
    LOG_DEBUG("cuImportExternalSemaphore");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuImportExternalSemaphore,extSem_out,semHandleDesc);
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
    LOG_DEBUG("cuSignalExternalSemaphoresAsync");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuSignalExternalSemaphoresAsync,extSemArray,paramsArray,numExtSems,stream);
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
    LOG_DEBUG("cuWaitExternalSemaphoresAsync");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuWaitExternalSemaphoresAsync,extSemArray,paramsArray,numExtSems,stream);
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
    LOG_DEBUG("cuDestroyExternalSemaphore");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDestroyExternalSemaphore,extSem);
}

/** Allocate asynchronously with per-thread semantics for a NULL stream. */
CUresult cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUstream stream) {
    return cuMemAllocAsync(dptr, bytesize, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}


/** Allocate from a pool with per-thread semantics for a NULL stream. */
CUresult cuMemAllocFromPoolAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream stream) {
    return cuMemAllocFromPoolAsync(dptr, bytesize, pool, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}


/** Free asynchronously with per-thread semantics for a NULL stream. */
CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream stream) {
    return cuMemFreeAsync(dptr, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}


/** Copy asynchronously with per-thread semantics for a NULL stream. */
CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t size, CUstream stream) {
    return cuMemcpyAsync(dst, src, size, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dst, const void *src, size_t size, CUstream stream) {
    return cuMemcpyHtoDAsync_v2(dst, src, size, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemcpyDtoHAsync_v2_ptsz(void *dst, CUdeviceptr src, size_t size, CUstream stream) {
    return cuMemcpyDtoHAsync_v2(dst, src, size, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t size, CUstream stream) {
    return cuMemcpyDtoDAsync_v2(dst, src, size, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemsetD8Async_ptsz(CUdeviceptr ptr, unsigned char value, size_t count, CUstream stream) {
    return cuMemsetD8Async(ptr, value, count, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemsetD2D8Async_ptsz(CUdeviceptr ptr, size_t pitch, unsigned char value, size_t width,
    size_t height, CUstream stream) {
    return cuMemsetD2D8Async(ptr, pitch, value, width, height,
        stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemsetD16Async_ptsz(CUdeviceptr ptr, uint16_t value, size_t count, CUstream stream) {
    return cuMemsetD16Async(ptr, value, count, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemsetD2D16Async_ptsz(CUdeviceptr ptr, size_t pitch, uint16_t value, size_t width, size_t height,
    CUstream stream) {
    return cuMemsetD2D16Async(ptr, pitch, value, width, height,
        stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemsetD32Async_ptsz(CUdeviceptr ptr, unsigned int value, size_t count, CUstream stream) {
    return cuMemsetD32Async(ptr, value, count, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuMemsetD2D32Async_ptsz(CUdeviceptr ptr, size_t pitch, unsigned int value, size_t width,
    size_t height, CUstream stream) {
    return cuMemsetD2D32Async(ptr, pitch, value, width, height,
        stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve per-thread stream semantics through the existing CUDA hook. */
CUresult cuLaunchKernel_ptsz(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
                              unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                              CUstream stream, void **params, void **extra) {
    return cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared,
        stream == NULL ? CU_STREAM_PER_THREAD : stream, params, extra);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *copy, CUstream stream) {
    return cuMemcpy2DAsync_v2(copy, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *copy, CUstream stream) {
    return cuMemcpy3DAsync_v2(copy, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *copy, CUstream stream) {
    return cuMemcpy3DPeerAsync(copy, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpyPeerAsync_ptsz(CUdeviceptr dst, CUcontext dst_ctx, CUdeviceptr src, CUcontext src_ctx,
                                 size_t size, CUstream stream) {
    return cuMemcpyPeerAsync(dst, dst_ctx, src, src_ctx, size,
        stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemPrefetchAsync_ptsz(CUdeviceptr ptr, size_t count, CUdevice dev, CUstream stream) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemPrefetchAsync_ptsz, ptr, count, dev,
        stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemPrefetchAsync_v2_ptsz(CUdeviceptr ptr, size_t count, CUmemLocation location, unsigned int flags,
                                    CUstream stream) {
    return cuMemPrefetchAsync_v2(ptr, count, location, flags,
        stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuLaunchCooperativeKernel_ptsz(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
                                        unsigned int bx, unsigned int by, unsigned int bz,
                                        unsigned int shared, CUstream stream, void **params) {
    return cuLaunchCooperativeKernel(function, gx, gy, gz, bx, by, bz, shared,
        stream == NULL ? CU_STREAM_PER_THREAD : stream, params);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src, size_t size) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpy_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpy_ptds, dst, src, size);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpyHtoD_v2_ptds(CUdeviceptr dst, const void *src, size_t size) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpyHtoD_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpyHtoD_v2_ptds, dst, src, size);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpyDtoH_v2_ptds(void *dst, CUdeviceptr src, size_t size) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpyDtoH_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpyDtoH_v2_ptds, dst, src, size);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpyDtoD_v2_ptds(CUdeviceptr dst, CUdeviceptr src, size_t size) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpyDtoD_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpyDtoD_v2_ptds, dst, src, size);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpyPeer_ptds(CUdeviceptr dst, CUcontext dst_ctx, CUdeviceptr src, CUcontext src_ctx, size_t size) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpyPeer_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpyPeer_ptds, dst, dst_ctx, src, src_ctx, size);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpy2D_v2_ptds(const CUDA_MEMCPY2D *copy) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpy2D_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpy2D_v2_ptds, copy);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *copy) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpy2DUnaligned_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpy2DUnaligned_v2_ptds, copy);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *copy) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpy3D_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpy3D_v2_ptds, copy);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *copy) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemcpy3DPeer_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpy3DPeer_ptds, copy);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemsetD8_v2_ptds(CUdeviceptr ptr, unsigned char value, size_t count) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemsetD8_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemsetD8_v2_ptds, ptr, value, count);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemsetD2D8_v2_ptds(CUdeviceptr ptr, size_t pitch, unsigned char value, size_t width, size_t height) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemsetD2D8_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemsetD2D8_v2_ptds, ptr, pitch, value, width, height);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemsetD16_v2_ptds(CUdeviceptr ptr, uint16_t value, size_t count) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemsetD16_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemsetD16_v2_ptds, ptr, value, count);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemsetD2D16_v2_ptds(CUdeviceptr ptr, size_t pitch, uint16_t value, size_t width, size_t height) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemsetD2D16_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemsetD2D16_v2_ptds, ptr, pitch, value, width, height);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemsetD32_v2_ptds(CUdeviceptr ptr, unsigned int value, size_t count) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemsetD32_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemsetD32_v2_ptds, ptr, value, count);
}

/** Preserve native per-thread stream behavior and return values. */
CUresult cuMemsetD2D32_v2_ptds(CUdeviceptr ptr, size_t pitch, unsigned int value, size_t width, size_t height) {
    ENSURE_RUNNING();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemsetD2D32_v2_ptds) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemsetD2D32_v2_ptds, ptr, pitch, value, width, height);
}

/** Copy the launch configuration while retaining the existing compute limiter. */
CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction function, void **params, void **extra) {
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuLaunchKernelEx) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    if (config == NULL) {
        if (CUDA_FIND_ENTRY(cuda_library_entry, cuLaunchKernelEx_ptsz) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
        return CUDA_OVERRIDE_CALL(cuda_library_entry, cuLaunchKernelEx_ptsz, config, function, params, extra);
    }
    CUlaunchConfig copy = *config;
    if (copy.hStream == NULL) copy.hStream = CU_STREAM_PER_THREAD;
    return cuLaunchKernelEx(&copy, function, params, extra);
}

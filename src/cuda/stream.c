#include "include/libcuda_hook.h"
#include "allocator/allocator.h"

CUresult cuStreamCreate(CUstream *phstream, unsigned int flags){
    LOG_INFO("cuStreamCreate %p",phstream);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamCreate,phstream,flags);
    return res;
}

CUresult cuStreamDestroy_v2 ( CUstream hStream ){
    LOG_DEBUG("cuStreamDestroy_v2 %p",hStream);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamDestroy_v2, hStream);
    if (res == CUDA_SUCCESS) graph_stream_destroyed(hStream);
    return res;
}

CUresult cuStreamSynchronize(CUstream hstream){
    LOG_DEBUG("cuStreamSync %p",hstream);
    uint64_t submitted = graph_sync_begin();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamSynchronize,hstream);
    if (res == CUDA_SUCCESS) graph_synchronized(NULL, hstream, 0, submitted);
    return res;
}

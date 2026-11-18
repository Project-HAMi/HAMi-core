#include "include/libcuda_hook.h"
#include "allocator/allocator.h"

CUresult cuStreamCreate(CUstream *phstream, unsigned int flags){
    LOG_INFO("cuStreamCreate %p",phstream);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamCreate,phstream,flags);
    return res;
}

/** Destroy a stream and retire its fallback graph-accounting identity on success. */
CUresult cuStreamDestroy_v2 ( CUstream hStream ){
    LOG_DEBUG("cuStreamDestroy_v2 %p",hStream);
    return graph_stream_destroy(hStream);
}

/** Synchronize a stream and release graph charges covered by the submission snapshot. */
CUresult cuStreamSynchronize(CUstream hstream){
    LOG_DEBUG("cuStreamSync %p",hstream);
    uint64_t submitted = graph_sync_begin();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamSynchronize,hstream);
    if (res == CUDA_SUCCESS) graph_synchronized(NULL, hstream, 0, submitted);
    return res;
}

/** Synchronize with per-thread semantics for a NULL stream. */
CUresult cuStreamSynchronize_ptsz(CUstream stream) {
    return cuStreamSynchronize(stream == NULL ? CU_STREAM_PER_THREAD : stream);
}

#include "include/libcuda_hook.h"
#include "allocator/allocator.h"
#include "multiprocess/multiprocess_memory_limit.h"

extern size_t context_size;
extern int ctx_activate[16];
void size_primary_context_lazily(int cudadev);

/* Serialize primary-context lifetimes with their bookkeeping per device. */
static pthread_mutex_t primary_context_mutex[CUDA_DEVICE_MAX_COUNT];
static pthread_once_t primary_context_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t legacy_context_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Guard context-handle reuse when unique context IDs are unavailable. */
static pthread_mutex_t *legacy_context_lock(void) {
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuCtxGetId) != NULL) return NULL;
    pthread_mutex_lock(&legacy_context_mutex);
    return &legacy_context_mutex;
}

static void primary_context_locks_init(void) {
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++)
        pthread_mutex_init(&primary_context_mutex[dev], NULL);
}

static pthread_mutex_t *primary_context_lock(CUdevice dev) {
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) return NULL;
    pthread_once(&primary_context_once, primary_context_locks_init);
    pthread_mutex_lock(&primary_context_mutex[dev]);
    return &primary_context_mutex[dev];
}

CUresult cuDevicePrimaryCtxGetState( CUdevice dev, unsigned int* flags, int* active ){
    LOG_DEBUG("into cuDevicePrimaryCtxGetState dev=%d",dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxGetState,dev,flags,active);
    return res;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev){
    pthread_mutex_t *legacy = legacy_context_lock();
    pthread_mutex_t *guard = primary_context_lock(dev);
    LOG_INFO("dev=%d context_size=%ld",dev,context_size);
    //for Initialization only
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRetain,pctx,dev);
    if (res == CUDA_SUCCESS) graph_primary_context(dev, *pctx);
    if (res == CUDA_SUCCESS && context_size == 0) {
        size_primary_context_lazily(dev);
    }
    if (ctx_activate[dev] == 0) {
        add_gpu_device_memory_usage(getpid(),dev,context_size,0); 
    }
    if (context_size>0) {
        ctx_activate[dev] = 1;
    }
    if (guard != NULL) pthread_mutex_unlock(guard);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}


CUresult cuDevicePrimaryCtxSetFlags_v2( CUdevice dev, unsigned int  flags ){
    LOG_DEBUG("into cuDevicePrimaryCtxSetFlags dev=%d flags=%d",dev,flags);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxSetFlags_v2,dev,flags);
}

CUresult cuDevicePrimaryCtxRelease_v2( CUdevice dev ){
    pthread_mutex_t *legacy = legacy_context_lock();
    pthread_mutex_t *guard = primary_context_lock(dev);
    if (ctx_activate[dev] == 1) {
        rm_gpu_device_memory_usage(getpid(),dev,context_size,0);
    }
    ctx_activate[dev] = 0;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRelease_v2,dev);
    if (res == CUDA_SUCCESS) {
        unsigned int flags;
        int active;
        if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuDevicePrimaryCtxGetState, dev,
                               &flags, &active) == CUDA_SUCCESS && !active)
            graph_primary_context_released(dev, 0);
    }
    if (guard != NULL) pthread_mutex_unlock(guard);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    pthread_mutex_t *legacy = legacy_context_lock();
    pthread_mutex_t *guard = primary_context_lock(dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDevicePrimaryCtxReset_v2, dev);
    if (res == CUDA_SUCCESS) graph_primary_context_released(dev, 1);
    if (guard != NULL) pthread_mutex_unlock(guard);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetDevice,device);
    return res;
}

#if CUDA_VERSION < 13000
CUresult cuCtxCreate_v2 ( CUcontext* pctx, unsigned int  flags, CUdevice dev ){
    pthread_mutex_t *legacy = legacy_context_lock();
    LOG_DEBUG("into cuCtxCreate pctx=%p flags=%d dev=%d",pctx,flags,dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxCreate_v2,pctx,flags,dev);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}

CUresult cuCtxCreate_v3 ( CUcontext* pctx, CUexecAffinityParam* paramsArray, int  numParams, unsigned int  flags, CUdevice dev ){
    pthread_mutex_t *legacy = legacy_context_lock();
    LOG_DEBUG("into cuCtxCreate_v3 pctx=%p paramsArray=%p numParams=%d flags=%d dev=%d",pctx,paramsArray,numParams,flags,dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxCreate_v3,pctx,paramsArray,numParams,flags,dev);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}
#endif

CUresult cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev) {
    pthread_mutex_t *legacy = legacy_context_lock();
    LOG_DEBUG("into cuCtxCreate_v4 pctx=%p ctxCreateParams=%p flags=%d dev=%d", pctx, ctxCreateParams, flags, dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxCreate_v4, pctx, ctxCreateParams, flags, dev);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}

CUresult cuCtxDestroy_v2 ( CUcontext ctx ){
    pthread_mutex_t *legacy = legacy_context_lock();
    cuuint64_t id = graph_context_id(ctx);
    LOG_DEBUG("into cuCtxDestroy_v2 ctx=%p",ctx);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxDestroy_v2, ctx);
    if (res == CUDA_SUCCESS) graph_context_destroyed(ctx, id);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}

CUresult cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version ){
    LOG_INFO("into cuCtxGetApiVersion ctx=%p",ctx);
    CUresult res =  CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetApiVersion,ctx,version);
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuCtxGetApiVersion res=%d",res);
    }
    return res;
}

CUresult cuCtxGetCacheConfig ( CUfunc_cache* pconfig ){
    LOG_DEBUG("into cuCtxGetCacheConfig");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetCacheConfig,pconfig);
}

CUresult cuCtxGetCurrent ( CUcontext* pctx ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetCurrent,pctx);
    return res;
}

CUresult cuCtxGetFlags ( unsigned int* flags ){
    LOG_DEBUG("into cuCtxGetFlags flags=%p",flags);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetFlags,flags);
}

CUresult cuCtxGetLimit ( size_t* pvalue, CUlimit limit ){
    LOG_DEBUG("into cuCtxGetLimit pvalue=%p",pvalue);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetLimit,pvalue,limit);
}

CUresult cuCtxGetSharedMemConfig ( CUsharedconfig* pConfig ){
    LOG_DEBUG("cuCtxGetSharedMemConfig pConfig=%p",pConfig);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetSharedMemConfig,pConfig);
}

CUresult cuCtxGetStreamPriorityRange ( int* leastPriority, int* greatestPriority ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetStreamPriorityRange,leastPriority,greatestPriority);
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuCtxGetStreamPriorityRange err=%d",res);
    }
    return res;
}

CUresult cuCtxPopCurrent_v2 ( CUcontext* pctx ){
    LOG_INFO("cuCtxPopCurrent pctx=%p",pctx);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxPopCurrent_v2,pctx);
}

CUresult cuCtxPushCurrent_v2 ( CUcontext ctx ){
    LOG_INFO("cuCtxPushCurrent ctx=%p",ctx);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxPushCurrent_v2,ctx);
}

CUresult cuCtxSetCacheConfig ( CUfunc_cache config ){
    LOG_DEBUG("cuCtxSetCacheConfig config=%d",config);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSetCacheConfig,config);
}

CUresult cuCtxSetCurrent ( CUcontext ctx ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSetCurrent,ctx);
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuCtxSetCurrent111 failed res=%d ctx=%p",res,ctx);
    }
    return res;
}

CUresult cuCtxSetLimit ( CUlimit limit, size_t value ){
    LOG_DEBUG("cuCtxSetLimit");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSetLimit,limit,value);
}

CUresult cuCtxSetSharedMemConfig ( CUsharedconfig config ){
    LOG_DEBUG("cuCtxSetSharedMemConfig");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSetSharedMemConfig,config);
}

CUresult cuCtxSynchronize ( void ){
    LOG_DEBUG("INTO CtxSync");
    uint64_t submitted = graph_sync_begin();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSynchronize);
    CUcontext ctx;
    if (res == CUDA_SUCCESS &&
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetCurrent, &ctx) == CUDA_SUCCESS)
        graph_synchronized(ctx, NULL, 1, submitted);
    return res;
}


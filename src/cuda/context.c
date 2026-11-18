#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "allocator/allocator.h"
#include "multiprocess/multiprocess_memory_limit.h"

extern size_t context_size;
extern int ctx_activate[16];
void size_primary_context_lazily(int cudadev);


/* Serialize primary-context lifetimes with their bookkeeping per device. */
static pthread_mutex_t primary_context_mutex[CUDA_DEVICE_MAX_COUNT];
static pthread_once_t primary_context_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t legacy_context_mutex = PTHREAD_MUTEX_INITIALIZER;

/** Serialize context reuse on drivers without native context IDs. */
static pthread_mutex_t *legacy_context_lock(void) {
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuCtxGetId) != NULL) return NULL;
    pthread_mutex_lock(&legacy_context_mutex);
    return &legacy_context_mutex;
}

/** Initialize the per-device primary-context lifetime mutexes. */
static void primary_context_locks_init(void) {
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++)
        pthread_mutex_init(&primary_context_mutex[dev], NULL);
}

/** Lock a valid device's primary-context lifetime bookkeeping. */
static pthread_mutex_t *primary_context_lock(CUdevice dev) {
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) return NULL;
    pthread_once(&primary_context_once, primary_context_locks_init);
    pthread_mutex_lock(&primary_context_mutex[dev]);
    return &primary_context_mutex[dev];
}

typedef struct {
    pthread_mutex_t *legacy;
    pthread_mutex_t *guard;
} primary_context_locks_t;

/** Take both lifetime locks in the fixed legacy-then-device order. */
static void lock_primary_context(CUdevice dev, primary_context_locks_t *locks) {
    locks->legacy = legacy_context_lock();
    locks->guard = primary_context_lock(dev);
}

/** pthread_cleanup handler, so a cancelled call does not keep the locks. */
static void unlock_primary_context(void *arg) {
    primary_context_locks_t *locks = arg;

    if (locks->guard != NULL) pthread_mutex_unlock(locks->guard);
    if (locks->legacy != NULL) pthread_mutex_unlock(locks->legacy);
}

static int primary_context_active(CUdevice dev) {
    unsigned int flags = 0;
    int active = 0;

    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDevicePrimaryCtxGetState,
                              dev, &flags, &active) == CUDA_SUCCESS && active;
}

CUresult cuDevicePrimaryCtxGetState( CUdevice dev, unsigned int* flags, int* active ){
    LOG_DEBUG("into cuDevicePrimaryCtxGetState dev=%d",dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxGetState,dev,flags,active);
    return res;
}

/** Retain a primary context and register its lifetime for graph accounting. */
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    ensure_hostpid();
    int lease_fd = -1;
    primary_context_locks_t locks;
    CUresult res;
    // Retaining an active context only takes a reference; skip the permit.
    // If another thread drops the last reference meanwhile, this retain
    // creates the context without a permit. Admission is best effort.
    // The permit comes before the locks, so no thread waits for the broker
    // while it holds a lock that other threads of this process need.
    if (driver_admission_enabled() && !primary_context_active(dev)) {
        lease_fd = driver_admission_acquire(HOSTPID_DRIVER_ADMISSION_RETAIN);
    }
    pthread_cleanup_push(driver_admission_cleanup, &lease_fd);
    lock_primary_context(dev, &locks);
    pthread_cleanup_push(unlock_primary_context, &locks);
    LOG_INFO("dev=%d context_size=%ld", dev, context_size);
    //for Initialization only
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRetain,pctx,dev);
    driver_admission_release(&lease_fd);
    if (res == CUDA_SUCCESS) {
        graph_primary_context(dev, *pctx);
    }
    if (res == CUDA_SUCCESS && context_size == 0) {
        size_primary_context_lazily(dev);
    }
    if (ctx_activate[dev] == 0) {
        add_gpu_device_memory_usage(getpid(),dev,context_size,0); 
    }
    if (context_size>0) {
        ctx_activate[dev] = 1;
    }
    pthread_cleanup_pop(1);
    pthread_cleanup_pop(1);
    return res;
}



CUresult cuDevicePrimaryCtxSetFlags_v2( CUdevice dev, unsigned int  flags ){
    LOG_DEBUG("into cuDevicePrimaryCtxSetFlags dev=%d flags=%d",dev,flags);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxSetFlags_v2,dev,flags);
}

/** Release a primary context and retire graph accounting once it becomes inactive. */
CUresult cuDevicePrimaryCtxRelease_v2( CUdevice dev ){
    int lease_fd = -1;
    primary_context_locks_t locks;
    CUresult res;
    // Wait for the permit first so the usage is not undercounted meanwhile.
    lease_fd = driver_admission_acquire(HOSTPID_DRIVER_ADMISSION_RELEASE);
    pthread_cleanup_push(driver_admission_cleanup, &lease_fd);
    lock_primary_context(dev, &locks);
    pthread_cleanup_push(unlock_primary_context, &locks);
    if (ctx_activate[dev] == 1) {
        rm_gpu_device_memory_usage(getpid(),dev,context_size,0);
    }
    ctx_activate[dev] = 0;
    graph_primary_context_prepare_release(dev);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRelease_v2,dev);
    driver_admission_release(&lease_fd);
    if (res == CUDA_SUCCESS) {
        unsigned int flags;
        int active;
        if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuDevicePrimaryCtxGetState, dev,
                               &flags, &active) == CUDA_SUCCESS && !active)
            graph_primary_context_released(dev, 0);
    }
    pthread_cleanup_pop(1);
    pthread_cleanup_pop(1);
    return res;
}


/** Reset a primary context and retire its previous graph-accounting generation. */
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    pthread_mutex_t *legacy = legacy_context_lock();
    pthread_mutex_t *guard = primary_context_lock(dev);
    graph_primary_context_prepare_release(dev);
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

/** Create a context while serializing handle reuse on drivers without context IDs. */
CUresult cuCtxCreate_v2 ( CUcontext* pctx, unsigned int  flags, CUdevice dev ){
    ensure_hostpid();
    pthread_mutex_t *legacy = legacy_context_lock();
    LOG_DEBUG("into cuCtxCreate pctx=%p flags=%d dev=%d", pctx, flags, dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxCreate_v2,pctx,flags,dev);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}


/** Create a context with affinity parameters while guarding legacy handle reuse. */
CUresult cuCtxCreate_v3 ( CUcontext* pctx, CUexecAffinityParam* paramsArray, int  numParams, unsigned int  flags, CUdevice dev ){
    ensure_hostpid();
    pthread_mutex_t *legacy = legacy_context_lock();
    LOG_DEBUG("into cuCtxCreate_v3 pctx=%p paramsArray=%p numParams=%d flags=%d dev=%d",
              pctx, paramsArray, numParams, flags, dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxCreate_v3,pctx,paramsArray,numParams,flags,dev);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}


#if CUDA_VERSION >= 12040
CUresult cuGreenCtxCreate(CUgreenCtx *phCtx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags) {
    LOG_DEBUG("into cuGreenCtxCreate dev=%d flags=%u", dev, flags);
    ensure_hostpid();
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGreenCtxCreate, phCtx, desc, dev, flags);
}

/** Retire completion tracking after a green context is destroyed. */
CUresult cuGreenCtxDestroy(CUgreenCtx green) {
    CUcontext ctx = NULL;
    CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxFromGreenCtx, &ctx, green);
    cuuint64_t id = graph_context_id(ctx);
    graph_context_prepare_destroy(ctx, id);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGreenCtxDestroy, green);
    if (res == CUDA_SUCCESS && ctx != NULL) graph_context_destroyed(ctx, id);
    return res;
}
#endif

/** Create a context with creation parameters while guarding legacy handle reuse. */
CUresult cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev) {
    ensure_hostpid();
    pthread_mutex_t *legacy = legacy_context_lock();
    LOG_DEBUG("into cuCtxCreate_v4 pctx=%p ctxCreateParams=%p flags=%d dev=%d", pctx, ctxCreateParams, flags, dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxCreate_v4, pctx, ctxCreateParams, flags, dev);
    if (legacy != NULL) pthread_mutex_unlock(legacy);
    return res;
}


/** Destroy a context and retire graph charges for its recorded generation. */
CUresult cuCtxDestroy_v2 ( CUcontext ctx ){
    pthread_mutex_t *legacy = legacy_context_lock();
    cuuint64_t id = graph_context_id(ctx);
    LOG_DEBUG("into cuCtxDestroy_v2 ctx=%p",ctx);
    graph_context_prepare_destroy(ctx, id);
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

/** Synchronize the current context and release charges covered by the submission snapshot. */
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

#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "cuda/context_accounting.h"
#include "multiprocess/multiprocess_memory_limit.h"

extern size_t context_size;
extern int ctx_activate[CUDA_DEVICE_MAX_COUNT];
extern int pidfound;

static size_t device_context_size[CUDA_DEVICE_MAX_COUNT];
static primary_context_accounting_t
    context_accounting[CUDA_DEVICE_MAX_COUNT];
static pthread_mutex_t context_accounting_lock = PTHREAD_MUTEX_INITIALIZER;

void context_accounting_fork_prepare() {
    pthread_mutex_lock(&context_accounting_lock);
}

void context_accounting_fork_parent() {
    pthread_mutex_unlock(&context_accounting_lock);
}

void context_accounting_fork_child() {
    int dev;

    context_size = 0;
    for (dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        context_accounting[dev].retain_count = 0;
        context_accounting[dev].charged_bytes = 0;
        device_context_size[dev] = 0;
        ctx_activate[dev] = 0;
    }
    pthread_mutex_unlock(&context_accounting_lock);
}


CUresult cuDevicePrimaryCtxGetState( CUdevice dev, unsigned int* flags, int* active ){
    LOG_DEBUG("into cuDevicePrimaryCtxGetState dev=%d",dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxGetState,dev,flags,active);
    return res;
}

static size_t context_charge_for_device(CUdevice dev) {
    if (dev >= 0 && dev < CUDA_DEVICE_MAX_COUNT &&
        device_context_size[dev] > 0) {
        return device_context_size[dev];
    }
    return context_size;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev){
    unsigned long long before = 0;
    unsigned long long after = 0;
    int hostpid;
    int measure_context = 0;
    size_t charge;
    size_t bytes_to_add = 0;

    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry,
                                  cuDevicePrimaryCtxRetain, pctx, dev);
    }

    pthread_mutex_lock(&context_accounting_lock);
    charge = context_charge_for_device(dev);
    hostpid = get_current_host_pid();
    if (charge == 0 && pidfound == 1 && hostpid > 0 &&
        context_accounting[dev].charged_bytes == 0) {
        nvmlReturn_t result = get_used_gpu_memory_by_pid(
            (unsigned int)hostpid, dev, &before);
        if (result == NVML_SUCCESS || result == NVML_ERROR_NOT_FOUND) {
            measure_context = 1;
            if (result == NVML_ERROR_NOT_FOUND) {
                before = 0;
            }
        }
    }

    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
                                      cuDevicePrimaryCtxRetain, pctx, dev);
    if (res != CUDA_SUCCESS) {
        pthread_mutex_unlock(&context_accounting_lock);
        return res;
    }

    if (measure_context) {
        int attempt;

        for (attempt = 0; attempt < 10; attempt++) {
            if (get_used_gpu_memory_by_pid((unsigned int)hostpid, dev,
                                           &after) == NVML_SUCCESS &&
                after > before) {
                device_context_size[dev] = after - before;
                charge = device_context_size[dev];
                LOG_INFO("Measured primary context size lazily: "
                         "dev=%d size=%lu", dev, charge);
                break;
            }
            usleep(1000);
        }
    }
    if (primary_context_record_retain(&context_accounting[dev], charge,
                                      &bytes_to_add) != 0) {
        LOG_ERROR("Primary context retain count overflow on device %d", dev);
    } else if (bytes_to_add > 0) {
        if (add_gpu_device_memory_usage(getpid(), dev, bytes_to_add, 0) != 0) {
            primary_context_cancel_charge(&context_accounting[dev]);
        }
    }
    ctx_activate[dev] = (int)context_accounting[dev].retain_count;
    if (context_accounting[dev].retain_count == 1 &&
        context_accounting[dev].charged_bytes == 0) {
        LOG_WARN("Primary context memory is not accounted for on device %d",
                 dev);
    }
    pthread_mutex_unlock(&context_accounting_lock);
    return res;
}


CUresult cuDevicePrimaryCtxSetFlags_v2( CUdevice dev, unsigned int  flags ){
    LOG_DEBUG("into cuDevicePrimaryCtxSetFlags dev=%d flags=%d",dev,flags);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxSetFlags_v2,dev,flags);
}

CUresult cuDevicePrimaryCtxRelease_v2( CUdevice dev ){
    size_t bytes_to_remove = 0;

    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry,
                                  cuDevicePrimaryCtxRelease_v2, dev);
    }
    pthread_mutex_lock(&context_accounting_lock);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRelease_v2,dev);
    if (res == CUDA_SUCCESS) {
        if (primary_context_record_release(&context_accounting[dev],
                                           &bytes_to_remove) != 0) {
            LOG_WARN("Unbalanced primary context release on device %d", dev);
        } else if (bytes_to_remove > 0) {
            rm_gpu_device_memory_usage(getpid(), dev, bytes_to_remove, 0);
        }
        ctx_activate[dev] = (int)context_accounting[dev].retain_count;
    }
    pthread_mutex_unlock(&context_accounting_lock);
    return res;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetDevice,device);
    return res;
}

#if CUDA_VERSION < 13000
CUresult cuCtxCreate_v2 ( CUcontext* pctx, unsigned int  flags, CUdevice dev ){
    LOG_DEBUG("into cuCtxCreate pctx=%p flags=%d dev=%d",pctx,flags,dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxCreate_v2,pctx,flags,dev);
    return res;
}

CUresult cuCtxCreate_v3 ( CUcontext* pctx, CUexecAffinityParam* paramsArray, int  numParams, unsigned int  flags, CUdevice dev ){
    LOG_DEBUG("into cuCtxCreate_v3 pctx=%p paramsArray=%p numParams=%d flags=%d dev=%d",pctx,paramsArray,numParams,flags,dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxCreate_v3,pctx,paramsArray,numParams,flags,dev);
    return res;
}
#endif

CUresult cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev) {
    LOG_DEBUG("into cuCtxCreate_v4 pctx=%p ctxCreateParams=%p flags=%d dev=%d", pctx, ctxCreateParams, flags, dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxCreate_v4, pctx, ctxCreateParams, flags, dev);
    return res;
}

CUresult cuCtxDestroy_v2 ( CUcontext ctx ){
    LOG_DEBUG("into cuCtxDestroy_v2 ctx=%p",ctx);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxDestroy_v2,ctx);
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
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSynchronize);
    return res;
}

#include <errno.h>

#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "cuda/context_accounting.h"
#include "multiprocess/multiprocess_memory_limit.h"

extern size_t context_size;
extern int pidfound;

static size_t device_context_size[CUDA_DEVICE_MAX_COUNT];
static primary_context_accounting_t
    context_accounting[CUDA_DEVICE_MAX_COUNT];
static pthread_mutex_t context_accounting_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t context_device_locks[CUDA_DEVICE_MAX_COUNT] = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
};

void context_accounting_fork_prepare() {
    pthread_mutex_lock(&context_accounting_lock);
}

void context_accounting_fork_parent() {
    pthread_mutex_unlock(&context_accounting_lock);
}

void context_accounting_fork_child() {
    int dev;

    /* context_size is kept.  The parent's probe is a fair estimate for a
     * child on the same GPU, and the child's own probe overwrites it. */
    primary_context_accounting_reset(context_accounting,
                                     CUDA_DEVICE_MAX_COUNT);
    for (dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        device_context_size[dev] = 0;
        context_device_locks[dev] =
            (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
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
    /* Device 0 trusts the host PID probe, which measured it.  Other devices
     * are measured on their first retain; if that fails, the retain path
     * falls back to the probed size rather than charging nothing. */
    return dev == 0 ? context_size : 0;
}

/* Only reached when the retain could not be recorded at all, so there is no
 * local accounting to undo: hand the context back and report the failure. */
static CUresult release_unaccounted_retain(CUdevice dev) {
    CUresult release_result;

    release_result = CUDA_OVERRIDE_CALL(cuda_library_entry,
                                        cuDevicePrimaryCtxRelease_v2, dev);
    if (release_result != CUDA_SUCCESS) {
        LOG_ERROR("Failed to release an unaccounted primary context on "
                  "device %d: %d", dev, release_result);
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev){
    uint64_t before = 0;
    uint64_t after = 0;
    int hostpid;
    int measure_context = 0;
    size_t charge;
    size_t measured_charge = 0;
    size_t bytes_to_add = 0;

    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry,
                                  cuDevicePrimaryCtxRetain, pctx, dev);
    }

    pthread_mutex_lock(&context_device_locks[dev]);
    pthread_mutex_lock(&context_accounting_lock);
    charge = context_charge_for_device(dev);
    hostpid = get_current_host_pid();
    if (charge == 0 && pidfound == 1 && hostpid > 0 &&
        context_accounting[dev].charged_bytes == 0) {
        measure_context = 1;
    }
    pthread_mutex_unlock(&context_accounting_lock);

    if (measure_context) {
        nvmlReturn_t result = get_used_gpu_memory_by_pid(
            (unsigned int)hostpid, dev, &before);
        if (result == NVML_SUCCESS || result == NVML_ERROR_NOT_FOUND) {
            if (result == NVML_ERROR_NOT_FOUND) {
                before = 0;
            }
        } else {
            measure_context = 0;
        }
    }

    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
                                      cuDevicePrimaryCtxRetain, pctx, dev);
    if (res != CUDA_SUCCESS) {
        pthread_mutex_unlock(&context_device_locks[dev]);
        return res;
    }

    if (measure_context) {
        int attempt;

        /* The delta is the process's own device memory across the retain, the
         * same method set_task_pid() uses.  context_device_locks[dev] does not
         * exclude cuMemAlloc, so an allocation by another thread of this
         * process inside the window is charged as context memory and cached
         * for the process lifetime.  Measuring once, on the first retain of a
         * device, keeps the exposure to that one window. */
        for (attempt = 0; attempt < 10; attempt++) {
            if (get_used_gpu_memory_by_pid((unsigned int)hostpid, dev,
                                           &after) == NVML_SUCCESS &&
                after > before) {
                measured_charge = after - before;
                break;
            }
            usleep(1000);
        }
    }
    pthread_mutex_lock(&context_accounting_lock);
    if (measured_charge > 0) {
        device_context_size[dev] = measured_charge;
        charge = measured_charge;
        LOG_INFO("Measured primary context size lazily: "
                 "dev=%d size=%lu", dev, charge);
    } else if (charge == 0 && context_size > 0) {
        /* Measurement failed or was skipped.  Charge the probed size, as main
         * does on every device.  Not cached in device_context_size, so a later
         * fresh retain can still measure this device for real. */
        charge = context_size;
        LOG_INFO("Primary context size unmeasured on device %d; charging the "
                 "probed size %lu", dev, charge);
    }
    errno = 0;
    int record_result =
        (pidfound == 1)
            ? primary_context_record_accounted_retain(
                  &context_accounting[dev], charge, &bytes_to_add)
            : primary_context_record_retain(&context_accounting[dev], charge,
                                            &bytes_to_add);
    /* The driver retain already succeeded.  An unknown context size must not
     * fail the caller, so defer the charge to a later retain that knows it. */
    if (record_result != 0 && errno == ENODATA) {
        LOG_WARN("Primary context size unknown on device %d; charge is "
                 "deferred to a later retain", dev);
        record_result = primary_context_record_retain(
            &context_accounting[dev], charge, &bytes_to_add);
    }
    if (record_result != 0) {
        LOG_ERROR("Cannot account primary context retain on device %d",
                  dev);
        res = release_unaccounted_retain(dev);
        pthread_mutex_unlock(&context_accounting_lock);
        pthread_mutex_unlock(&context_device_locks[dev]);
        return res;
    }
    if (bytes_to_add > 0 &&
        add_gpu_device_memory_usage(getpid(), dev, bytes_to_add, 0) != 0) {
        size_t retried = 0;

        /* The driver retain already succeeded.  A shared region that will not
         * take the charge must not fail the caller, for the same reason an
         * unknown size does not: drop the charge, keep the retain, and let a
         * later retain try again. */
        LOG_WARN("Cannot charge primary context memory on device %d; the "
                 "retain is kept and the charge is deferred", dev);
        if (primary_context_rollback_retain(&context_accounting[dev],
                                            bytes_to_add) != 0 ||
            primary_context_record_retain(&context_accounting[dev], 0,
                                          &retried) != 0) {
            LOG_ERROR("Cannot reconcile context accounting on device %d", dev);
        }
    }
    pthread_mutex_unlock(&context_accounting_lock);
    pthread_mutex_unlock(&context_device_locks[dev]);
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
    pthread_mutex_lock(&context_device_locks[dev]);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRelease_v2,dev);
    if (res == CUDA_SUCCESS) {
        pthread_mutex_lock(&context_accounting_lock);
        if (primary_context_record_release(&context_accounting[dev],
                                           &bytes_to_remove) != 0) {
            LOG_WARN("Unbalanced primary context release on device %d", dev);
        } else if (bytes_to_remove > 0) {
            if (rm_gpu_device_memory_usage(getpid(), dev, bytes_to_remove,
                                           0) != 0) {
                primary_context_restore_charge(&context_accounting[dev],
                                               bytes_to_remove);
            }
        }
        pthread_mutex_unlock(&context_accounting_lock);
    }
    pthread_mutex_unlock(&context_device_locks[dev]);
    return res;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    size_t bytes_to_remove = 0;

    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry,
                                  cuDevicePrimaryCtxReset_v2, dev);
    }
    pthread_mutex_lock(&context_device_locks[dev]);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
                                      cuDevicePrimaryCtxReset_v2, dev);
    if (res == CUDA_SUCCESS) {
        pthread_mutex_lock(&context_accounting_lock);
        /* The driver destroys the context whatever the retain count, so
         * every outstanding retain and the charge go with it. */
        bytes_to_remove = context_accounting[dev].charged_bytes;
        primary_context_accounting_reset(&context_accounting[dev], 1);
        if (bytes_to_remove > 0 &&
            rm_gpu_device_memory_usage(getpid(), dev, bytes_to_remove,
                                       0) != 0) {
            primary_context_restore_charge(&context_accounting[dev],
                                           bytes_to_remove);
        }
        pthread_mutex_unlock(&context_accounting_lock);
    }
    pthread_mutex_unlock(&context_device_locks[dev]);
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

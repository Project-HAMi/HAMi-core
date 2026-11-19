#include "allocator/allocator.h"
#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "include/memory_limit.h"

CUresult cuEventCreate ( CUevent* phEvent, unsigned int  Flags ){
    LOG_DEBUG("cuEventCreate Event=%p",phEvent);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuEventCreate,phEvent,Flags);
}

CUresult cuEventDestroy_v2 ( CUevent hEvent ){
    LOG_DEBUG("cuEventDestroy_v2 hEvent=%p",hEvent);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuEventDestroy_v2,hEvent);
}

/* The driver allocates module globals itself when it loads a module, so
 * cuMemAlloc never sees them. Charge what each load takes as module memory
 * (type 1) and give it back when the module is unloaded. */
typedef struct module_charge {
    CUmodule module;
    CUdevice dev;
    size_t size;
    struct module_charge *next;
} module_charge_t;

typedef struct {
    CUdevice dev;
    int64_t untracked;
} module_sample_t;

static pthread_mutex_t module_lock = PTHREAD_MUTEX_INITIALIZER;
static module_charge_t *module_charges;

/** Device memory of this process that HAMi has not charged: what NVML reports
 *  for the process minus its allocations. Only run while none is in flight. */
static int sample_untracked(void *arg) {
    module_sample_t *s = arg;
    uint64_t used = 0, data = 0;
    int32_t hostpid = 0;

    if (get_own_device_usage(s->dev, &hostpid, &data) != 0 ||
        get_used_gpu_memory_by_pid(hostpid, s->dev, &used) != NVML_SUCCESS) {
        return -1;
    }
    s->untracked = (int64_t)used - (int64_t)data;
    return 0;
}

/** Lock module bookkeeping and sample memory before a load. */
static int module_load_begin(module_sample_t *before) {
    pthread_mutex_lock(&module_lock);
    if (cuCtxGetDevice(&before->dev) != CUDA_SUCCESS ||
        with_memory_settled(sample_untracked, before) != 0) {
        LOG_WARN("module load not measured, its memory stays uncharged");
        return -1;
    }
    return 0;
}

/** Charge the memory a successful load took, then unlock. */
static CUresult module_load_end(CUresult res, CUmodule *module, int measured,
                                const module_sample_t *before) {
    module_sample_t after = {.dev = before->dev, .untracked = 0};
    module_charge_t *charge = NULL;
    size_t size;

    if (res != CUDA_SUCCESS || measured != 0 || module == NULL) {
        goto out;
    }
    if (with_memory_settled(sample_untracked, &after) != 0) {
        LOG_WARN("module load not measured, its memory stays uncharged");
        goto out;
    }
    if (after.untracked <= before->untracked) {
        goto out;
    }
    size = (size_t)(after.untracked - before->untracked);
    charge = malloc(sizeof(*charge));
    if (charge == NULL) {
        goto out;
    }
    add_gpu_device_memory_usage(getpid(), before->dev, size, 1);
    charge->module = *module;
    charge->dev = before->dev;
    charge->size = size;
    charge->next = module_charges;
    module_charges = charge;
    LOG_INFO("module %p on dev %d charged %zu bytes",
             (void *)charge->module, charge->dev, charge->size);
out:
    pthread_mutex_unlock(&module_lock);
    return res;
}

CUresult cuModuleLoad ( CUmodule* module, const char* fname ){
    module_sample_t before = {0, 0};
    LOG_DEBUG(" cuModuleLoad fname=%s",fname);
    int measured = module_load_begin(&before);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoad, module, fname);
    return module_load_end(res, module, measured, &before);
}

CUresult cuModuleLoadData( CUmodule* module, const void* image){
    module_sample_t before = {0, 0};
    LOG_DEBUG("cuModuleLoadData module=%p",module);
    int measured = module_load_begin(&before);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadData, module, image);
    return module_load_end(res, module, measured, &before);
}

CUresult cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues ){
    module_sample_t before = {0, 0};
    LOG_DEBUG("cuModuleLoadDataEx module=%p",module);
    int measured = module_load_begin(&before);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadDataEx, module, image,
                                      numOptions, options, optionValues);
    return module_load_end(res, module, measured, &before);
}

CUresult cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin ){
    module_sample_t before = {0, 0};
    LOG_DEBUG("cuModuleLoadFatBinary module=%p",module);
    int measured = module_load_begin(&before);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadFatBinary, module, fatCubin);
    return module_load_end(res, module, measured, &before);
}

CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleGetFunction,hfunc,hmod,name);
    return res;
}

CUresult cuModuleUnload(CUmodule hmod) {
    module_charge_t **link, *charge;
    LOG_DEBUG("cuModuleUnload");
    pthread_mutex_lock(&module_lock);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleUnload, hmod);
    if (res == CUDA_SUCCESS) {
        for (link = &module_charges; *link != NULL; link = &(*link)->next) {
            if ((*link)->module == hmod) {
                charge = *link;
                *link = charge->next;
                lock_shrreg();
                rm_gpu_device_memory_usage(getpid(), charge->dev, charge->size, 1);
                unlock_shrreg();
                free(charge);
                break;
            }
        }
    }
    pthread_mutex_unlock(&module_lock);
    return res;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleGetGlobal_v2,dptr,bytes,hmod,name);
    return res;
}

CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
    LOG_INFO("cuModuleGetTexRef");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleGetTexRef,pTexRef,hmod,name);
}

CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name) {
    LOG_INFO("cuModuleGetSurfRef");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleGetSurfRef,pSurfRef,hmod,name);
}

CUresult cuLinkAddData_v2 ( CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues ) {
    LOG_DEBUG("into cuLinkAddData_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuLinkAddData_v2,state,type,data,size,name,numOptions,options,optionValues);
}

CUresult cuLinkCreate_v2 ( unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut ) {
    LOG_DEBUG("into cuLinkCreate_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuLinkCreate_v2,numOptions,options,optionValues,stateOut);
}

CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path,
    unsigned int numOptions, CUjit_option *options, void **optionValues) {
    LOG_DEBUG("cuLinkAddFile_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuLinkAddFile_v2,state,type,path,numOptions,options,optionValues);
    }

CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    LOG_DEBUG("cuLinkComplete");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuLinkComplete,state,cubinOut,sizeOut);
}

CUresult cuLinkDestroy(CUlinkState state) {
    LOG_DEBUG("cuLinkDestroy");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuLinkDestroy,state);
}

CUresult cuFuncSetCacheConfig ( CUfunction hfunc, CUfunc_cache config ){
    LOG_INFO("cuFUncSetCacheConfig");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuFuncSetCacheConfig,hfunc,config);
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
    LOG_INFO("cuFuncSetSharedMemConfig");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuFuncSetSharedMemConfig,hfunc,config);
}

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
    LOG_DEBUG("cuFuncGetAttribute");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuFuncGetAttribute,pi,attrib,hfunc);
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    LOG_DEBUG("cuFuncSetAttribute");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuFuncSetAttribute,hfunc,attrib,value);
}

#include "include/libcuda_hook.h"
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

static pthread_mutex_t module_lock = PTHREAD_MUTEX_INITIALIZER;
static module_charge_t *module_charges;

/** Device memory in use that HAMi does not track. Allocations HAMi counts
 *  while a module loads change both sides, so they cancel out. */
static int untracked_device_memory(CUdevice dev, int64_t *untracked) {
    size_t drv_free = 0, drv_total = 0;

    if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemGetInfo_v2,
                           &drv_free, &drv_total) != CUDA_SUCCESS) {
        return -1;
    }
    *untracked = (int64_t)(drv_total - drv_free) -
                 (int64_t)get_current_device_memory_usage(cuda_to_nvml_map(dev));
    return 0;
}

/** Lock module bookkeeping and note untracked memory before a load. */
static int module_load_begin(CUdevice *dev, int64_t *before) {
    pthread_mutex_lock(&module_lock);
    if (cuCtxGetDevice(dev) != CUDA_SUCCESS) {
        return -1;
    }
    return untracked_device_memory(*dev, before);
}

/** Charge the memory a successful load took, then unlock. */
static void module_load_end(CUresult res, const CUmodule *module, CUdevice dev,
                            int measured, int64_t before) {
    int64_t after;
    module_charge_t *charge;

    if (res == CUDA_SUCCESS && measured == 0 && module != NULL &&
        untracked_device_memory(dev, &after) == 0 && after > before) {
        charge = malloc(sizeof(*charge));
        if (charge != NULL) {
            charge->module = *module;
            charge->dev = dev;
            charge->size = (size_t)(after - before);
            charge->next = module_charges;
            module_charges = charge;
            add_gpu_device_memory_usage(getpid(), dev, charge->size, 1);
            LOG_INFO("module %p on dev %d charged %zu bytes",
                     (void *)charge->module, dev, charge->size);
        }
    }
    pthread_mutex_unlock(&module_lock);
}

CUresult cuModuleLoad ( CUmodule* module, const char* fname ){
    CUdevice dev = 0;
    int64_t before = 0;
    LOG_DEBUG(" cuModuleLoad fname=%s",fname);
    int measured = module_load_begin(&dev, &before);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleLoad,module,fname);
    module_load_end(res, module, dev, measured, before);
    return res;
}

CUresult cuModuleLoadData( CUmodule* module, const void* image){
    CUdevice dev = 0;
    int64_t before = 0;
    LOG_DEBUG("cuModuleLoadData module=%p",module);
    int measured = module_load_begin(&dev, &before);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleLoadData,module,image);
    module_load_end(res, module, dev, measured, before);
    return res;
}

CUresult cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues ){
    CUdevice dev = 0;
    int64_t before = 0;
    LOG_DEBUG("cuModuleLoadDataEx module=%p",module);
    int measured = module_load_begin(&dev, &before);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleLoadDataEx,module,image,
                                      numOptions,options,optionValues);
    module_load_end(res, module, dev, measured, before);
    return res;
}

CUresult cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin ){
    CUdevice dev = 0;
    int64_t before = 0;
    LOG_DEBUG("cuModuleLoadFatBinary module=%p",module);
    int measured = module_load_begin(&dev, &before);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleLoadFatBinary,module,fatCubin);
    module_load_end(res, module, dev, measured, before);
    return res;
}

CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleGetFunction,hfunc,hmod,name);
    return res;
}

CUresult cuModuleUnload(CUmodule hmod) {
    module_charge_t **link, *charge;
    LOG_DEBUG("cuModuleUnload");
    pthread_mutex_lock(&module_lock);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleUnload,hmod);
    if (res == CUDA_SUCCESS) {
        for (link = &module_charges; *link != NULL; link = &(*link)->next) {
            if ((*link)->module == hmod) {
                charge = *link;
                *link = charge->next;
                rm_gpu_device_memory_usage(getpid(), charge->dev, charge->size, 1);
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

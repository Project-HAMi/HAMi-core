#include <string.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "include/memory_limit.h"
#include "include/utils.h"

typedef struct {
    CUcontext ctx;
    cuuint64_t ctx_id;
    CUdevice dev;
    int64_t before;
    int sampled;
} module_load_scope_t;

static int sample_unaccounted_self_bytes(CUdevice dev, int64_t *out) {
    int32_t hostpid = 0;
    uint64_t tracked = 0;
    uint64_t nvml_used = 0;
    int ret = -1;
    memory_sampling_enter();
    if (get_self_device_memory_snapshot(dev, &hostpid, &tracked) == 0 &&
        get_process_gpu_memory(hostpid, dev, &nvml_used) == 0) {
        *out = (int64_t)nvml_used - (int64_t)tracked;
        ret = 0;
    }
    memory_sampling_leave();
    return ret;
}

static void begin_module_load(module_load_scope_t *scope) {
    memset(scope, 0, sizeof(*scope));
    scope->dev = -1;
    ENSURE_RUNNING();
    module_load_lock();
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuCtxGetCurrent) != NULL &&
        CUDA_FIND_ENTRY(cuda_library_entry, cuCtxGetDevice) != NULL &&
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetCurrent, &scope->ctx) == CUDA_SUCCESS &&
        scope->ctx != NULL &&
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetDevice, &scope->dev) == CUDA_SUCCESS) {
        scope->ctx_id = graph_context_id(scope->ctx);
        if (sample_unaccounted_self_bytes(scope->dev, &scope->before) == 0) {
            scope->sampled = 1;
        }
    }
}

static CUresult finish_module_load(const module_load_scope_t *scope,
                                   CUmodule *module, CUresult res) {
    if (res != CUDA_SUCCESS || module == NULL || *module == NULL || !scope->sampled) {
        module_load_unlock();
        return res;
    }
    int64_t after = 0;
    size_t size = 0;
    if (sample_unaccounted_self_bytes(scope->dev, &after) == 0) {
        if (after > scope->before) {
            size = (size_t)(after - scope->before);
        }
    } else {
        LOG_WARN("failed to sample post-load GPU memory on dev=%d", scope->dev);
    }
    if (size > 0) {
        CUresult admit_res = module_admit_loaded(*module, scope->ctx,
                                                 scope->ctx_id, scope->dev, size);
        if (admit_res != CUDA_SUCCESS) {
            if (CUDA_FIND_ENTRY(cuda_library_entry, cuModuleUnload) != NULL) {
                CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleUnload, *module);
            }
            *module = NULL;
            module_load_unlock();
            return admit_res;
        }
    }
    module_load_unlock();
    return CUDA_SUCCESS;
}

CUresult cuEventCreate ( CUevent* phEvent, unsigned int  Flags ){
    LOG_DEBUG("cuEventCreate Event=%p",phEvent);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuEventCreate,phEvent,Flags);
}

CUresult cuEventDestroy_v2 ( CUevent hEvent ){
    LOG_DEBUG("cuEventDestroy_v2 hEvent=%p",hEvent);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuEventDestroy_v2,hEvent);
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    LOG_DEBUG("cuModuleLoad fname=%s", fname);
    module_load_scope_t scope;
    begin_module_load(&scope);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoad, module, fname);
    return finish_module_load(&scope, module, res);
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    LOG_DEBUG("cuModuleLoadData module=%p", module);
    module_load_scope_t scope;
    begin_module_load(&scope);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadData, module, image);
    return finish_module_load(&scope, module, res);
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                            CUjit_option *options, void **optionValues) {
    LOG_DEBUG("cuModuleLoadDataEx module=%p", module);
    module_load_scope_t scope;
    begin_module_load(&scope);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadDataEx, module, image,
                                      numOptions, options, optionValues);
    return finish_module_load(&scope, module, res);
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    LOG_DEBUG("cuModuleLoadFatBinary module=%p", module);
    module_load_scope_t scope;
    begin_module_load(&scope);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadFatBinary, module, fatCubin);
    return finish_module_load(&scope, module, res);
}

CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleGetFunction,hfunc,hmod,name);
    return res;
}

CUresult cuModuleUnload(CUmodule hmod) {
    LOG_DEBUG("cuModuleUnload");
    module_load_lock();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleUnload, hmod);
    if (res == CUDA_SUCCESS && hmod != NULL) {
        module_release_unloaded(hmod);
    }
    module_load_unlock();
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

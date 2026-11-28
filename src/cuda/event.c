#include <pthread.h>
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
    int locked;
    int graph_quiesced;
} module_load_scope_t;

static void module_load_scope_cleanup(void *arg) {
    module_load_scope_t *scope = (module_load_scope_t *)arg;
    if (scope->graph_quiesced) {
        module_graph_quiesce_leave();
        scope->graph_quiesced = 0;
    }
    if (scope->locked) {
        module_load_unlock();
        scope->locked = 0;
    }
}

static void unlock_module_load_cleanup(void *unused) {
    (void)unused;
    module_load_unlock();
}

static void leave_memory_sampling_cleanup(void *unused) {
    (void)unused;
    memory_sampling_leave();
}

static int sample_unaccounted_self_bytes(CUdevice dev, int64_t *out) {
    int32_t hostpid = 0;
    uint64_t tracked = 0;
    uint64_t nvml_used = 0;
    int ret = -1;
    memory_sampling_enter();
    pthread_cleanup_push(leave_memory_sampling_cleanup, NULL);
    if (get_self_device_memory_snapshot(dev, &hostpid, &tracked) == 0 &&
        get_process_gpu_memory(hostpid, dev, &nvml_used) == 0) {
        *out = (int64_t)nvml_used - (int64_t)tracked;
        ret = 0;
    }
    pthread_cleanup_pop(1);
    return ret;
}

static void begin_module_load(module_load_scope_t *scope) {
    memset(scope, 0, sizeof(*scope));
    scope->dev = -1;
    ENSURE_RUNNING();
    module_load_lock();
    scope->locked = 1;
    module_graph_quiesce_enter();
    scope->graph_quiesced = 1;
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

static CUresult finish_module_load(module_load_scope_t *scope,
                                   CUmodule *module, CUresult res) {
    if (res != CUDA_SUCCESS || module == NULL || *module == NULL || !scope->sampled) {
        module_load_scope_cleanup(scope);
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
    if (scope->graph_quiesced) {
        module_graph_quiesce_leave();
        scope->graph_quiesced = 0;
    }
    if (size > 0) {
        CUresult admit_res = module_admit_loaded(*module, scope->ctx,
                                                 scope->ctx_id, scope->dev, size);
        if (admit_res != CUDA_SUCCESS) {
            if (CUDA_FIND_ENTRY(cuda_library_entry, cuModuleUnload) != NULL) {
                CUresult unload_res = CUDA_OVERRIDE_CALL(cuda_library_entry,
                                                         cuModuleUnload, *module);
                if (unload_res != CUDA_SUCCESS) {
                    LOG_WARN("failed to roll back rejected module load: %d", unload_res);
                }
            }
            *module = NULL;
            module_load_scope_cleanup(scope);
            return admit_res;
        }
    }
    module_load_scope_cleanup(scope);
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
    module_load_scope_t scope = {0};
    CUresult res;
    pthread_cleanup_push(module_load_scope_cleanup, &scope);
    begin_module_load(&scope);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoad, module, fname);
    res = finish_module_load(&scope, module, res);
    pthread_cleanup_pop(1);
    return res;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    LOG_DEBUG("cuModuleLoadData module=%p", module);
    module_load_scope_t scope = {0};
    CUresult res;
    pthread_cleanup_push(module_load_scope_cleanup, &scope);
    begin_module_load(&scope);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadData, module, image);
    res = finish_module_load(&scope, module, res);
    pthread_cleanup_pop(1);
    return res;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                            CUjit_option *options, void **optionValues) {
    LOG_DEBUG("cuModuleLoadDataEx module=%p", module);
    module_load_scope_t scope = {0};
    CUresult res;
    pthread_cleanup_push(module_load_scope_cleanup, &scope);
    begin_module_load(&scope);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadDataEx, module, image,
                             numOptions, options, optionValues);
    res = finish_module_load(&scope, module, res);
    pthread_cleanup_pop(1);
    return res;
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    LOG_DEBUG("cuModuleLoadFatBinary module=%p", module);
    module_load_scope_t scope = {0};
    CUresult res;
    pthread_cleanup_push(module_load_scope_cleanup, &scope);
    begin_module_load(&scope);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleLoadFatBinary, module, fatCubin);
    res = finish_module_load(&scope, module, res);
    pthread_cleanup_pop(1);
    return res;
}

CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleGetFunction,hfunc,hmod,name);
    return res;
}

CUresult cuModuleUnload(CUmodule hmod) {
    LOG_DEBUG("cuModuleUnload");
    CUresult res;
    module_load_lock();
    pthread_cleanup_push(unlock_module_load_cleanup, NULL);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuModuleUnload, hmod);
    if (res == CUDA_SUCCESS && hmod != NULL) {
        module_release_unloaded(hmod);
    }
    pthread_cleanup_pop(1);
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

#include "include/libcuda_hook.h"
#include <nvml.h>

CUresult cuEventCreate ( CUevent* phEvent, unsigned int  Flags ){
    LOG_DEBUG("cuEventCreate Event=%p",phEvent);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuEventCreate,phEvent,Flags);
}

CUresult cuEventDestroy_v2 ( CUevent hEvent ){
    LOG_DEBUG("cuEventDestroy_v2 hEvent=%p",hEvent);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuEventDestroy_v2,hEvent);
}

CUresult cuModuleLoad ( CUmodule* module, const char* fname ){
    LOG_DEBUG(" cuModuleLoad fname=%s",fname);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleLoad,module,fname);
}

CUresult cuModuleLoadData( CUmodule* module, const void* image){
    LOG_DEBUG("cuModuleLoadData module=%p",module);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleLoadData,module,image);
}

CUresult cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues ){
    LOG_DEBUG("cuModuleLoadDataEx module=%p",module);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleLoadDataEx,module,image,numOptions,options,optionValues);
}

CUresult cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin ){
    LOG_DEBUG("cuModuleLoadFatBinary module=%p",module);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleLoadFatBinary,module,fatCubin);
}

CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleGetFunction,hfunc,hmod,name);
    return res;
}

CUresult cuModuleUnload(CUmodule hmod) {
    LOG_DEBUG("cuModuleUnload");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuModuleUnload,hmod);
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

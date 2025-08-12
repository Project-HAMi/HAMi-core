#include <string.h>
#include <ctype.h>
#include <dlfcn.h>
#include "include/libnvml_hook.h"
#include "include/nvml-subset.h"
#include "include/utils.h"
#include "multiprocess/multiprocess_memory_limit.h"

entry_t nvml_library_entry[]= {
    #define X(func) {.name = #func, .fn_ptr = NULL},
    NVML_FUNCTIONS(X)
    #undef X
};

_Static_assert(sizeof(nvml_library_entry)/sizeof(entry_t) == NVML_ENTRY_END,
               "nvml_library_entry size doesn't match nvml_override_enum_t");

pthread_once_t init_virtual_map_pre_flag = PTHREAD_ONCE_INIT;
pthread_once_t init_virtual_map_post_flag = PTHREAD_ONCE_INIT;

typedef void* (*fp_dlsym)(void*, const char*);
extern fp_dlsym real_dlsym;
extern int virtual_nvml_devices;
extern int cuda_to_nvml_map_array[16];

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    return NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetIndex, device, index);
}


extern void* _dl_sym(void*, const char*, void*);

void load_nvml_libraries() {
    void *table = NULL;
    char driver_filename[FILENAME_MAX];

    if (real_dlsym == NULL) {
        real_dlsym = dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
        if (real_dlsym == NULL) {
            real_dlsym = _dl_sym(RTLD_NEXT, "dlsym", dlsym);
            if (real_dlsym == NULL)
                LOG_ERROR("real dlsym not found");
        }
    }
    snprintf(driver_filename, FILENAME_MAX - 1, "%s", "libnvidia-ml.so.1");
    driver_filename[FILENAME_MAX - 1] = '\0';

    table = dlopen(driver_filename, RTLD_NOW | RTLD_NODELETE);
    if (!table) {
        LOG_WARN("can't find library %s", driver_filename);  
    }
    int i;
    for (i = 0; i < NVML_ENTRY_END; i++) {
        LOG_DEBUG("loading %s:%d",nvml_library_entry[i].name,i);
        nvml_library_entry[i].fn_ptr = real_dlsym(table, nvml_library_entry[i].name);
        if (!nvml_library_entry[i].fn_ptr) {
            LOG_INFO("can't find function %s in %s", nvml_library_entry[i].name,
                driver_filename);
        }
    }
    LOG_INFO("loaded nvml libraries");
    dlclose(table);
}

void nvml_preInit() {
    ensure_initialized();
    load_env_from_file(ENV_OVERRIDE_FILE);
    load_nvml_libraries();
    int i;
    for (i=0; i<16; i++) {
        cuda_to_nvml_map_array[i] = i;
    }   
}

void nvml_postInit() {
    init_device_info();
}

nvmlReturn_t _nvmlDeviceGetMemoryInfo(nvmlDevice_t device,void* memory,int version) {
    unsigned int dev_id;
    LOG_DEBUG("into nvmlDeviceGetMemoryInfo");

    switch (version){
        case 1:
            CHECK_NVML_API(NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetMemoryInfo, device, memory));
            LOG_DEBUG("origin_free=%lld total=%lld\n", ((nvmlMemory_t*)memory)->free, ((nvmlMemory_t*)memory)->total);
            break;
        case 2:
            CHECK_NVML_API(NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetMemoryInfo_v2, device, (nvmlMemory_v2_t *)memory));
            LOG_DEBUG("origin_free=%lld total=%lld\n", ((nvmlMemory_v2_t*)memory)->free, ((nvmlMemory_v2_t*)memory)->total);
    }
    CHECK_NVML_API(nvmlDeviceGetIndex(device, &dev_id));
    int cudadev = nvml_to_cuda_map(dev_id);
    if (cudadev < 0)
        return NVML_SUCCESS;
    size_t usage = get_current_device_memory_usage(cudadev);
    size_t monitor = get_current_device_memory_monitor(cudadev);
    size_t limit = get_current_device_memory_limit(cudadev);
    LOG_DEBUG("usage=%ld limit=%ld monitor=%ld",usage,limit,monitor);
    if ( memory == NULL) {
        return NVML_SUCCESS;
    }
    if (limit == 0){
        switch (version){
        case 1:
             ((nvmlMemory_t*)memory)->used = usage;
            return NVML_SUCCESS;
        case 2:
            ((nvmlMemory_v2_t *)memory)->used = usage;
            return NVML_SUCCESS;
        }
    } else {
        switch (version){
        case 1:
             ((nvmlMemory_t*)memory)->free = (limit-usage);
             ((nvmlMemory_t*)memory)->total = limit;
             ((nvmlMemory_t*)memory)->used = usage;
            return NVML_SUCCESS;
        case 2:
            ((nvmlMemory_v2_t *)memory)->free = (limit-usage);
            ((nvmlMemory_v2_t *)memory)->total = limit;
            ((nvmlMemory_v2_t *)memory)->used = usage;
            return NVML_SUCCESS;
        } 
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    return _nvmlDeviceGetMemoryInfo(device,memory,1); 
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) {
    return _nvmlDeviceGetMemoryInfo(device,memory,2);
}


nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2 ( nvmlDevice_t device, unsigned int  link, nvmlPciInfo_t* pci ) {
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetNvLinkRemotePciInfo_v2,device,link,pci);
    return res;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo ( nvmlDevice_t device, unsigned int  link, nvmlPciInfo_t* pci ) {
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetNvLinkRemotePciInfo,device,link,pci);
    return res;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex ( unsigned int  index, nvmlDevice_t* device ){
    nvmlReturn_t res;
    LOG_DEBUG("nvmlDeviceGetHandleByIndex index=%u",index); 
    res = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry,nvmlDeviceGetHandleByIndex,index,device);
    return res;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2 ( unsigned int  index, nvmlDevice_t* device ){
    nvmlReturn_t res;
    LOG_DEBUG("nvmlDeviceGetHandleByIndex_v2 index=%u",index); 
    res = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry,nvmlDeviceGetHandleByIndex_v2,index,device);
    return res;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2 ( const char* pciBusId, nvmlDevice_t* device ) {
    LOG_INFO("NVML DeviceGetHandleByPciBusID_v2 %s",pciBusId);
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetHandleByPciBusId_v2,pciBusId,device);
}


nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId,
                                           nvmlDevice_t *device) {
    LOG_DEBUG("NVML DeviceGetHandleByPciBusId %s",pciBusId);
    return NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetHandleByPciBusId,
                         pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetHandleBySerial ( const char* serial, nvmlDevice_t* device ) {
    LOG_INFO("NVML DeviceGetHandleBySerial Not supported %s",serial);
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetHandleBySerial,serial,device);
}

nvmlReturn_t nvmlDeviceGetHandleByUUID ( const char* uuid, nvmlDevice_t* device ) {
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetHandleByUUID,uuid,device);
    return res;
}

nvmlReturn_t nvmlDeviceGetCount ( unsigned int* deviceCount ) {
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetCount_v2,deviceCount);
}

nvmlReturn_t nvmlDeviceGetCount_v2 ( unsigned int* deviceCount ) {
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetCount_v2,deviceCount);
}

nvmlReturn_t nvmlInitWithFlags( unsigned int  flags ) {
    LOG_DEBUG("nvmlInitWithFlags")
    pthread_once(&init_virtual_map_pre_flag, (void(*) (void))nvml_preInit);
    nvmlReturn_t res =  NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInitWithFlags,flags);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlInit(void) {
    LOG_DEBUG("nvmlInit")
    pthread_once(&init_virtual_map_pre_flag,(void (*)(void))nvml_preInit);
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInit_v2);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlInit_v2(void) {
    LOG_DEBUG("nvmlInit_v2");
    pthread_once(&init_virtual_map_pre_flag,(void (*)(void))nvml_preInit);
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInit_v2);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetPciInfo_v3, device,
                         pci);
  return res;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetPciInfo_v2, device,
                         pci);
  return res;
}

nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  nvmlReturn_t res =  NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetPciInfo, device, pci);
  return res;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid,
                               unsigned int length) {
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetUUID, device, uuid,
                         length);
    return res;
}

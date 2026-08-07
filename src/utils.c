#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <sys/file.h>
#include "include/utils.h"
#include "include/log_utils.h"
#include "include/nvml_prefix.h"
#include <nvml.h>
#include "include/nvml_override.h"
#include "include/libcuda_hook.h"
#include "include/hostpid_broker.h"
#include "multiprocess/multiprocess_memory_limit.h"

const char* unified_lock="/tmp/vgpulock/lock";
static int lock_fd = -1;
extern size_t context_size;
extern int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];
extern int cuda_to_nvml_map_count;
static nvmlReturn_t map_cuda_devices_to_nvml_by_pci(void);

// 0 unified_lock lock success
// -1 unified_lock lock fail
int try_lock_unified_lock() {
    lock_fd = open(unified_lock, O_CREAT | O_RDWR, 0666);
    if (lock_fd == -1) {
        LOG_ERROR("failed to open unified_lock file: %s", unified_lock);
        return -1;
    }
    if (flock(lock_fd, LOCK_EX) == -1) {
        LOG_ERROR("flock failed on unified_lock");
        close(lock_fd);
        lock_fd = -1;
        return -1;
    }
    LOG_INFO("try_lock_unified_lock: acquired");
    return 0;
}

// 0 unified_lock unlock success
// -1 unified_lock unlock fail
int try_unlock_unified_lock() {
    if (lock_fd == -1) {
        LOG_ERROR("try_unlock_unified_lock: no lock held");
        return -1;
    }
    int res = flock(lock_fd, LOCK_UN);
    close(lock_fd);
    lock_fd = -1;
    LOG_INFO("try unlock_unified_lock:%d", res);
    return res == 0 ? 0 : -1;
}

int mergepid(unsigned int *prev, unsigned int *current, nvmlProcessInfo_t1 *sub, nvmlProcessInfo_t1 *merged) {
    int i,j;
    int found=0;
    for (i=0;i<*prev;i++){
        found=0;
        for (j=0;j<*current;j++) {
            LOG_INFO("merge pid=%d",sub[i].pid);
            if (sub[i].pid == merged[j].pid) {
                found = 1;
                break;
            } 
        }
        if (!found) {
            LOG_DEBUG("merged pid=%d\n",sub[i].pid);
            merged[*current] = sub[i];
            (*current)++;
        }
    }
    return 0;
}

int getextrapid(unsigned int prev, unsigned int current, nvmlProcessInfo_t1 *pre_pids_on_device, nvmlProcessInfo_t1 *pids_on_device) {
    int i,j;
    int found = 0;
    for (i=0; i<prev; i++){
        LOG_INFO("prev pids[%d]=%d",i,pre_pids_on_device[i].pid);
    }
    for (i=0; i< current; i++) {
        LOG_INFO("current pids[%d]=%d",i,pids_on_device[i].pid);
    }
    if (current-prev<=0)
        return 0;
    for (i=0; i<current; i++) {
        found = 0;
        for (j=0; j<prev; j++) {
            if (pids_on_device[i].pid == pre_pids_on_device[j].pid) {
                found = 1;
                break;
            }
        }
        if (!found)
            return pids_on_device[i].pid;
    }
    return 0;
}

nvmlReturn_t set_task_pid_from_broker() {
    const char *enabled = getenv("LIBVGPU_HOSTPID_BROKER");
    nvmlReturn_t result;
    pid_t hostpid = 0;

    if (!hostpid_broker_enabled(enabled)) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    if (hostpid_broker_query_trusted(HOSTPID_BROKER_SOCKET_PATH,
                                     &hostpid) != 0) {
        LOG_DEBUG("Host PID broker unavailable: %s", strerror(errno));
        return NVML_ERROR_NOT_FOUND;
    }
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        LOG_WARN("NVML initialization failed after broker lookup: %d",
                 result);
        return result;
    }
    result = map_cuda_devices_to_nvml_by_pci();
    if (result != NVML_SUCCESS) {
        LOG_WARN("CUDA to NVML PCI mapping failed after broker lookup: %d",
                 result);
        return result;
    }
    if (set_host_pid(hostpid) != 0) {
        return NVML_ERROR_NOT_FOUND;
    }

    LOG_INFO("Host PID %d read from trusted broker", hostpid);
    return NVML_SUCCESS;
}

nvmlReturn_t get_used_gpu_memory_by_pid(unsigned int process_pid, int cudadev,
                                        unsigned long long *used) {
    nvmlProcessInfo_v1_t *processes = NULL;
    nvmlDevice_t device;
    nvmlReturn_t result;
    unsigned int count = SHARED_REGION_MAX_PROCESS_NUM;
    unsigned int i;
    int nvmldev;
    unsigned int mapped_dev;

    if (used == NULL || process_pid == 0 || cudadev < 0 ||
        cudadev >= CUDA_DEVICE_MAX_COUNT) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *used = 0;
    mapped_dev = cuda_to_nvml_map((unsigned int)cudadev);
    if (mapped_dev >= CUDA_DEVICE_MAX_COUNT) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    nvmldev = (int)mapped_dev;

    result = nvmlDeviceGetHandleByIndex(nvmldev, &device);
    if (result != NVML_SUCCESS) {
        return result;
    }

    processes = calloc(count, sizeof(*processes));
    if (processes == NULL) {
        return NVML_ERROR_MEMORY;
    }
    result = nvmlDeviceGetComputeRunningProcesses(device, &count, processes);
    if (result == NVML_ERROR_INSUFFICIENT_SIZE && count > 0) {
        if ((size_t)count > SIZE_MAX / sizeof(*processes)) {
            free(processes);
            return NVML_ERROR_MEMORY;
        }
        nvmlProcessInfo_v1_t *larger =
            realloc(processes, count * sizeof(*processes));
        if (larger == NULL) {
            free(processes);
            return NVML_ERROR_MEMORY;
        }
        processes = larger;
        result =
            nvmlDeviceGetComputeRunningProcesses(device, &count, processes);
    }
    if (result != NVML_SUCCESS) {
        free(processes);
        return result;
    }

    result = NVML_ERROR_NOT_FOUND;
    for (i = 0; i < count; i++) {
        if (processes[i].pid == process_pid &&
            processes[i].usedGpuMemory != NVML_VALUE_NOT_AVAILABLE) {
            *used = processes[i].usedGpuMemory;
            result = NVML_SUCCESS;
            break;
        }
    }
    free(processes);
    return result;
}

nvmlReturn_t set_task_pid() {
    unsigned int running_processes=0,previous=0,merged_num=0;
    nvmlProcessInfo_v1_t tmp_pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlProcessInfo_t1 pre_pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlProcessInfo_t1 pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlDevice_t device;
    nvmlReturn_t res;
    CUcontext pctx;
    int i;
    CHECK_NVML_API(nvmlInit());
    CHECK_NVML_API(nvmlDeviceGetHandleByIndex(0, &device));
    
    unsigned int nvmlCounts;
    CHECK_NVML_API(nvmlDeviceGetCount(&nvmlCounts));
    
    int cudaDev;
    for (i=0;i<nvmlCounts;i++){
        cudaDev=nvml_to_cuda_map(i);
        if (cudaDev<0) {
            continue;
        }
        CHECK_NVML_API(nvmlDeviceGetHandleByIndex(i, &device));
        do{
            res = nvmlDeviceGetComputeRunningProcesses(device, &previous, tmp_pids_on_device);
            if ((res != NVML_SUCCESS) && (res != NVML_ERROR_INSUFFICIENT_SIZE)) {
                LOG_ERROR("Device2GetComputeRunningProcesses failed %d,%d\n",res,i);
                return res;
            }
        }while(res==NVML_ERROR_INSUFFICIENT_SIZE); 
        mergepid(&previous,&merged_num,(nvmlProcessInfo_t1 *)tmp_pids_on_device,pre_pids_on_device);
        break;
    }
    previous = merged_num;
    merged_num = 0;
    memset(tmp_pids_on_device,0,sizeof(nvmlProcessInfo_v1_t)*SHARED_REGION_MAX_PROCESS_NUM);
    CHECK_CU_RESULT(cuDevicePrimaryCtxRetain(&pctx,0));
    for (i=0;i<nvmlCounts;i++) {
        cudaDev=nvml_to_cuda_map(i);
        if (cudaDev<0) {
            continue;
        }
        CHECK_NVML_API(nvmlDeviceGetHandleByIndex (i, &device)); 
        do{
            res = nvmlDeviceGetComputeRunningProcesses(device, &running_processes, tmp_pids_on_device);
            if ((res != NVML_SUCCESS) && (res != NVML_ERROR_INSUFFICIENT_SIZE)) {
                LOG_ERROR("Device2GetComputeRunningProcesses failed %d\n",res);
                return res;
            }
        }while(res == NVML_ERROR_INSUFFICIENT_SIZE);
        mergepid(&running_processes,&merged_num,(nvmlProcessInfo_t1 *)tmp_pids_on_device,pids_on_device);
        break;
    }
    running_processes = merged_num;
    LOG_INFO("current processes num = %u %u",previous,running_processes);
    for (i=0;i<merged_num;i++){
        LOG_INFO("current pid in use is %d %d",i,pids_on_device[i].pid);
        //tmp_pids_on_device[i].pid=0;
    }
    unsigned int hostpid = getextrapid(previous,running_processes,pre_pids_on_device,pids_on_device); 
    if (hostpid==0) {
        LOG_ERROR("host pid is error!");
        return NVML_ERROR_DRIVER_NOT_LOADED;
    }
    LOG_INFO("hostPid=%d",hostpid);
    if (set_host_pid(hostpid)==0) {
        for (i=0;i<running_processes;i++) {
            if (pids_on_device[i].pid==hostpid) {
                unsigned long long measured =
                    pids_on_device[i].usedGpuMemory;

                if (measured == NVML_VALUE_NOT_AVAILABLE ||
                    (unsigned long long)(size_t)measured != measured) {
                    context_size = 0;
                    LOG_WARN("Primary context memory is unavailable");
                } else {
                    context_size = (size_t)measured;
                    LOG_INFO("Primary Context Size==%llu", measured);
                }
                break;
            }
        }
    }
    CHECK_CU_RESULT(cuDevicePrimaryCtxRelease(0));
    return NVML_SUCCESS; 
}

int parse_cuda_visible_env() {
    char *s = getenv("CUDA_VISIBLE_DEVICES");
    int count = 0;
    for (int i = 0; i < CUDA_DEVICE_MAX_COUNT; i++) {
        cuda_to_nvml_map_array[i] = i;
    }

    if (need_cuda_virtualize()) {
        for (int i = 0; i < strlen(s); i++) {
            if ((s[i] == ',') || (i == 0)) {
                int tmp = (i==0) ? atoi(s) : atoi(s + i +1);
                cuda_to_nvml_map_array[count] = tmp; 
                count++;
            }
        } 
    }
    for (int i = 0; i < CUDA_DEVICE_MAX_COUNT; i++) {
        LOG_INFO("device %d -> %d",i,cuda_to_nvml_map(i));
    }
    LOG_INFO("get default cuda from %s", getenv("CUDA_VISIBLE_DEVICES"));
    return count;
}

int map_cuda_visible_devices() {
    int visible_count = 0;

    parse_cuda_visible_env();
    if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetCount,
                           &visible_count) == CUDA_SUCCESS &&
        visible_count >= 0 &&
        visible_count <= CUDA_DEVICE_MAX_COUNT) {
        cuda_to_nvml_map_count = visible_count;
    }
    return 0;
}

static nvmlReturn_t map_cuda_devices_to_nvml_by_pci(void) {
    unsigned int mapped_devices[CUDA_DEVICE_MAX_COUNT];
    int cuda_device_count = 0;
    int ordinal;
    CUresult cuda_result;

    cuda_result = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetCount,
                                     &cuda_device_count);
    if (cuda_result != CUDA_SUCCESS || cuda_device_count <= 0 ||
        cuda_device_count > CUDA_DEVICE_MAX_COUNT) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    for (ordinal = 0; ordinal < cuda_device_count; ordinal++) {
        char pci_bus_id[32] = {0};
        CUdevice cuda_device;
        nvmlDevice_t nvml_device;
        nvmlReturn_t result;

        cuda_result = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGet,
                                         &cuda_device, ordinal);
        if (cuda_result != CUDA_SUCCESS) {
            return NVML_ERROR_NOT_FOUND;
        }
        cuda_result = CUDA_OVERRIDE_CALL(cuda_library_entry,
                                         cuDeviceGetPCIBusId,
                                         pci_bus_id,
                                         (int)sizeof(pci_bus_id),
                                         cuda_device);
        if (cuda_result != CUDA_SUCCESS) {
            return NVML_ERROR_NOT_FOUND;
        }
        result = nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id,
                                                   &nvml_device);
        if (result != NVML_SUCCESS) {
            return result;
        }
        result = nvmlDeviceGetIndex(nvml_device,
                                    &mapped_devices[ordinal]);
        if (result != NVML_SUCCESS) {
            return result;
        }
        if (mapped_devices[ordinal] >= CUDA_DEVICE_MAX_COUNT) {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    for (ordinal = 0; ordinal < CUDA_DEVICE_MAX_COUNT; ordinal++) {
        cuda_to_nvml_map_array[ordinal] = CUDA_DEVICE_MAX_COUNT;
    }
    for (ordinal = 0; ordinal < cuda_device_count; ordinal++) {
        cuda_to_nvml_map_array[ordinal] = (int)mapped_devices[ordinal];
        LOG_INFO("CUDA device %d maps to NVML device %u by PCI identity",
                 ordinal, mapped_devices[ordinal]);
    }
    cuda_to_nvml_map_count = cuda_device_count;
    return NVML_SUCCESS;
}

int getenvcount() {
    char *s = getenv("CUDA_VISIBLE_DEVICES");
    if ((s == NULL) || (strlen(s)==0)){
        return -1;
    }
    LOG_DEBUG("get from env %s",s);
    int i,count=0;
    for (i=0;i<strlen(s);i++){
        if (s[i]==',')
            count++;
    }
    return count+1;
}

int need_cuda_virtualize() {
    int count1 = -1;
    char *s = getenv("CUDA_VISIBLE_DEVICES");
    if ((s == NULL) || (strlen(s)==0)){
        return 0;
    }
    int fromenv = getenvcount();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetCount,&count1);
    if (res != CUDA_SUCCESS) {
        return 1;
    }
    LOG_DEBUG("count1=%d",count1);
    if (fromenv ==count1) {
        return 1;
    }
    return 0;
}

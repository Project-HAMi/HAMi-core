#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <time.h>
#include <errno.h>
#include <sys/file.h>
#include "include/utils.h"
#include "include/log_utils.h"
#include "include/nvml_prefix.h"
#include <nvml.h>
#include "include/nvml_override.h"
#include "include/libcuda_hook.h"
#include "include/hostpid_broker.h"
#include "include/hostpid_fallback_lock.h"
#include "multiprocess/multiprocess_memory_limit.h"

extern size_t context_size;
extern int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

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
            merged[*current].pid = sub[i].pid;
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
    if (current <= prev)
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

static nvmlReturn_t get_used_gpu_memory_by_pid(pid_t hostpid,
                                              unsigned long long *used) {
    nvmlProcessInfo_v1_t *processes;
    nvmlDevice_t device;
    nvmlReturn_t res;
    unsigned int count = SHARED_REGION_MAX_PROCESS_NUM;
    unsigned int i;

    res = nvmlDeviceGetHandleByIndex(cuda_to_nvml_map(0), &device);
    if (res != NVML_SUCCESS) {
        return res;
    }
    processes = calloc(count, sizeof(*processes));
    if (processes == NULL) {
        return NVML_ERROR_MEMORY;
    }
    res = nvmlDeviceGetComputeRunningProcesses(device, &count, processes);
    if (res == NVML_SUCCESS) {
        res = NVML_ERROR_NOT_FOUND;
        for (i = 0; i < count; i++) {
            if (processes[i].pid == (unsigned int)hostpid &&
                processes[i].usedGpuMemory !=
                    (unsigned long long)NVML_VALUE_NOT_AVAILABLE) {
                *used = processes[i].usedGpuMemory;
                res = NVML_SUCCESS;
                break;
            }
        }
    }
    free(processes);
    return res;
}

// The broker names this process's host PID without NVML discovery. Context
// sizing shares the node lock so its retain cannot disturb a fallback probe.
nvmlReturn_t set_task_pid_from_broker(void) {
    pid_t hostpid = 0;
    unsigned long long used = 0;
    CUcontext pctx;
    nvmlReturn_t res;
    struct timespec deadline;

    if (hostpid_broker_query_trusted(HOSTPID_BROKER_SOCKET_PATH,
                                    &hostpid) != 0) {
        LOG_WARN("Host PID broker unavailable: %s", strerror(errno));
        return NVML_ERROR_UNKNOWN;
    }
    LOG_INFO("hostPid=%d from broker", hostpid);
    if (set_host_pid(hostpid) != 0) {
        return NVML_ERROR_NOT_FOUND;
    }
    // The PID is already set, so a sizing failure only leaves context_size
    // alone, as a missing NVML entry does in set_task_pid().
    if (nvmlInit() != NVML_SUCCESS) {
        LOG_WARN("Primary context size unavailable");
        return NVML_SUCCESS;
    }
    if (hostpid_fallback_lock_deadline_after_ms(
            &deadline, HOSTPID_FALLBACK_LOCK_TIMEOUT_MS) != 0 ||
        hostpid_fallback_lock_acquire_shared_until(&deadline) != 0) {
        LOG_WARN("Skipped primary context sizing because the node lock "
                 "failed: %s", strerror(errno));
        return NVML_SUCCESS;
    }
    if (cuDevicePrimaryCtxRetain(&pctx, 0) != CUDA_SUCCESS) {
        LOG_WARN("Primary context size unavailable");
        goto unlock;
    }
    res = get_used_gpu_memory_by_pid(hostpid, &used);
    if (res == NVML_SUCCESS) {
        LOG_INFO("Primary Context Size==%llu", used);
        context_size = used;
    } else {
        LOG_WARN("Primary context size unavailable: %d", res);
    }
    if (cuDevicePrimaryCtxRelease(0) != CUDA_SUCCESS) {
        LOG_WARN("Failed to release the device 0 primary context");
    }
unlock:
    if (hostpid_fallback_lock_release() != 0) {
        LOG_ERROR("Failed to release the context sizing lock: %s",
                  strerror(errno));
    }
    return NVML_SUCCESS;
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
                LOG_INFO("Primary Context Size==%lld",tmp_pids_on_device[i].usedGpuMemory);
                context_size = tmp_pids_on_device[i].usedGpuMemory; 
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
                if (count >= CUDA_DEVICE_MAX_COUNT) {
                    LOG_ERROR("CUDA_VISIBLE_DEVICES exceeds max count");
                    break;
                }
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
    parse_cuda_visible_env();
    return 0;
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

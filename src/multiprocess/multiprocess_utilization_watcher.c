#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include <cuda.h>
#include "include/nvml_prefix.h"
#include <nvml.h>
#include <sys/time.h>
#include <sys/wait.h>

#include "multiprocess/multiprocess_memory_limit.h"
#include "multiprocess/multiprocess_utilization_watcher.h"
#include "include/log_utils.h"
#include "include/nvml_override.h"


static int g_sm_num;
static int g_max_thread_per_sm;
static volatile long g_cur_cuda_cores = 0;
static volatile long g_total_cuda_cores = 0;
extern int pidfound;
int cuda_to_nvml_map_array[16];

void rate_limiter(int grids, int blocks) {
  long before_cuda_cores = 0;
  long after_cuda_cores = 0;
  long kernel_size = grids;

  while (get_recent_kernel()<0) {
    sleep(1);
  }
  set_recent_kernel(2);
  if ((get_current_device_sm_limit(0)>=100) || (get_current_device_sm_limit(0)==0))
    	return;
  if (get_utilization_switch()==0)
      return;
  LOG_DEBUG("grid: %d, blocks: %d", grids, blocks);
  LOG_DEBUG("launch kernel %ld, curr core: %ld", kernel_size, g_cur_cuda_cores);
  //if (g_vcuda_config.enable) {
    do {
CHECK:
      before_cuda_cores = g_cur_cuda_cores;
      LOG_DEBUG("current core: %ld", g_cur_cuda_cores);
      if (before_cuda_cores < 0) {
        nanosleep(&g_cycle, NULL);
        goto CHECK;
      }
      after_cuda_cores = before_cuda_cores - kernel_size;
    } while (!CAS(&g_cur_cuda_cores, before_cuda_cores, after_cuda_cores));
  //}
}

static void change_token(long delta) {
  int cuda_cores_before = 0, cuda_cores_after = 0;

  LOG_DEBUG("delta: %ld, curr: %ld", delta, g_cur_cuda_cores);
  do {
    cuda_cores_before = g_cur_cuda_cores;
    cuda_cores_after = cuda_cores_before + delta;

    if (cuda_cores_after > g_total_cuda_cores) {
      cuda_cores_after = g_total_cuda_cores;
    }
  } while (!CAS(&g_cur_cuda_cores, cuda_cores_before, cuda_cores_after));
}

long delta(int up_limit, int user_current, long share) {
  int utilization_diff =
      abs(up_limit - user_current) < 5 ? 5 : abs(up_limit - user_current);
  long increment =
      (long)g_sm_num * (long)g_sm_num * (long)g_max_thread_per_sm * (long)utilization_diff / 2560;

  /* Accelerate cuda cores allocation when utilization vary widely */
  if (utilization_diff > up_limit / 2) {
    increment = increment * utilization_diff * 2 / (up_limit + 1);
  }

  if (user_current <= up_limit) {
    share = (share + increment) > g_total_cuda_cores ? g_total_cuda_cores
                                                   : (share + increment);
  } else {
    share = (share - increment) < 0 ? 0 : (share - increment);
  }

  return share;
}

unsigned int nvml_to_cuda_map(unsigned int nvmldev){
    unsigned int devcount;
    CHECK_NVML_API(nvmlDeviceGetCount_v2(&devcount));
    int i=0;
    for (i=0;i<devcount;i++){
        if (cuda_to_nvml_map(i)==nvmldev)
          return i;
    }
    return -1;
}

unsigned int cuda_to_nvml_map(unsigned int cudadev){
    return cuda_to_nvml_map_array[cudadev];
}

int setspec() {
    CHECK_NVML_API(nvmlInit());
    CHECK_CU_RESULT(cuDeviceGetAttribute(&g_sm_num,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,0));
    CHECK_CU_RESULT(cuDeviceGetAttribute(&g_max_thread_per_sm,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,0));
    g_total_cuda_cores = g_max_thread_per_sm * g_sm_num * FACTOR;
    return 0;
}

int get_used_gpu_utilization(int *userutil,int *sysprocnum) {
    struct timeval cur;
    size_t microsec;

    int i,sum=0;
    unsigned int infcount;
    nvmlProcessInfo_v1_t infos[SHARED_REGION_MAX_PROCESS_NUM];

    unsigned int nvmlCounts;
    CHECK_NVML_API(nvmlDeviceGetCount(&nvmlCounts));
    lock_shrreg();

    int devi,cudadev;
    for (devi=0;devi<nvmlCounts;devi++){
      sum=0;
      infcount = SHARED_REGION_MAX_PROCESS_NUM;
      shrreg_proc_slot_t *proc;
      cudadev = nvml_to_cuda_map((unsigned int)(devi));
      if (cudadev<0)
        continue;
      userutil[cudadev] = 0;
      nvmlDevice_t device;
      CHECK_NVML_API(nvmlDeviceGetHandleByIndex(cudadev, &device));

      //Get Memory for container
      nvmlReturn_t res = nvmlDeviceGetComputeRunningProcesses(device,&infcount,infos);
      if (res == NVML_SUCCESS) {
        for (i=0; i<infcount; i++){
          proc = find_proc_by_hostpid(infos[i].pid);
          if (proc != NULL){
              //LOG_DEBUG("pid=%u monitor=%lld\n", infos[i].pid, infos[i].usedGpuMemory);
              proc->monitorused[cudadev] += infos[i].usedGpuMemory;
          }
        }
      }
      // Get SM util for container

      gettimeofday(&cur,NULL);
      microsec = (cur.tv_sec - 1) * 1000UL * 1000UL + cur.tv_usec;
      nvmlProcessUtilizationSample_t processes_sample[SHARED_REGION_MAX_PROCESS_NUM];
      unsigned int processes_num = SHARED_REGION_MAX_PROCESS_NUM;
      res = nvmlDeviceGetProcessUtilization(device,processes_sample,&processes_num,microsec);
      if (res == NVML_SUCCESS) {
        for (i=0; i<processes_num; i++){
          proc = find_proc_by_hostpid(processes_sample[i].pid);
          if (proc != NULL){
              sum += processes_sample[i].smUtil;
              //LOG_DEBUG("pid=%u smUtil=%d\n", processes_sample[i].pid, processes_sample[i].smUtil);
              proc->device_util[cudadev].sm_util += processes_sample[i].smUtil;
          }
        }
      }
      if (sum < 0)
        sum = 0;
      userutil[cudadev] = sum;
      unlock_shrreg();
    }
    unlock_shrreg();
    return 0;
}

void* utilization_watcher() {
    nvmlInit();
    int userutil[CUDA_DEVICE_MAX_COUNT];
    int sysprocnum;
    long share = 0;
    int upper_limit = get_current_device_sm_limit(0);
    ensure_initialized();
    LOG_DEBUG("upper_limit=%d\n",upper_limit);
    while (1){
        nanosleep(&g_wait, NULL);
        if (pidfound==0) {
          update_host_pid();
          if (pidfound==0)
            continue;
        }
        init_gpu_device_utilization();
        get_used_gpu_utilization(userutil,&sysprocnum);
        //if (sysprocnum == 1 &&
        //    userutil < upper_limit / 10) {
        //    g_cur_cuda_cores =
        //        delta(upper_limit, userutil, share);
        //    continue;
        //}
        if ((share==g_total_cuda_cores) && (g_cur_cuda_cores<0)) {
          g_total_cuda_cores *= 2;
          share = g_total_cuda_cores;
        }
        if ((userutil[0]<=100) && (userutil[0]>=0)){
          share = delta(upper_limit, userutil[0], share);
          change_token(share);
        }
        LOG_INFO("userutil1=%d currentcores=%ld total=%ld limit=%d share=%ld\n",userutil[0],g_cur_cuda_cores,g_total_cuda_cores,upper_limit,share);
    }
}

void init_utilization_watcher() {
    LOG_INFO("set core utilization limit to  %d",get_current_device_sm_limit(0));
    setspec();
    pthread_t tid;
    if ((get_current_device_sm_limit(0)<=100) && (get_current_device_sm_limit(0)>0)){
        pthread_create(&tid, NULL, utilization_watcher, NULL);
    }
    return;
}

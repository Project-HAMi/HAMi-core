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

#include <limits.h>

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

  // Prevent overflow by breaking down the calculation and adding bounds checking
  long increment = 0;

  if (g_sm_num > 0 && g_max_thread_per_sm > 0 && utilization_diff > 0) {
    // Calculate in steps to avoid overflow
    increment = (long)g_sm_num * (long)utilization_diff / 256;
    increment = increment * ((long)g_sm_num * (long)g_max_thread_per_sm / 10);

    // Ensure increment is positive and not too large
    if (increment < 0) {
      increment = 0;
    } else if (increment > g_total_cuda_cores) {
      increment = g_total_cuda_cores / 2;  // Limit to a reasonable value
    }

    /* Accelerate cuda cores allocation when utilization vary widely */
    if (utilization_diff > up_limit / 2 && up_limit > 0) {
      long accel_factor = (long)utilization_diff * 2 / (up_limit + 1);
      // Prevent overflow in multiplication
      if (accel_factor > 0 && increment <= LONG_MAX / accel_factor) {
        increment = increment * accel_factor;
      }
    }
  }

  // Update share with bounds checking
  if (user_current <= up_limit) {
    // Check for overflow before adding
    if (increment > LONG_MAX - share) {
      share = g_total_cuda_cores;
    } else {
      share = (share + increment) > g_total_cuda_cores ? g_total_cuda_cores : (share + increment);
    }
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
      uint64_t sum=0;
      uint64_t usedGpuMemory=0;
      infcount = SHARED_REGION_MAX_PROCESS_NUM;
      shrreg_proc_slot_t *proc = NULL;
      shrreg_proc_slot_t *self_proc = NULL;
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
              usedGpuMemory += infos[i].usedGpuMemory;
              // Store our own process for later use
              if (proc->pid == getpid()) {
                  self_proc = proc;
              }
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
              // Store our own process for later use
              if (proc->pid == getpid()) {
                  self_proc = proc;
              }
          }
        }
      }
      if (sum < 0)
        sum = 0;
      if (usedGpuMemory < 0)
        usedGpuMemory = 0;

      // Use self_proc instead of potentially invalid proc pointer
      if (self_proc != NULL) {
          self_proc->device_util[cudadev].sm_util = sum;
          self_proc->monitorused[cudadev] = usedGpuMemory;
      }
      userutil[cudadev] = sum;
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

    // Track consecutive failures to find processes
    int consecutive_empty_processes = 0;
    const int max_empty_retries = 5;

    while (1){
        nanosleep(&g_wait, NULL);
        if (pidfound==0) {
          update_host_pid();
          if (pidfound==0) {
            // Don't immediately give up if we can't find processes
            consecutive_empty_processes++;
            if (consecutive_empty_processes > max_empty_retries) {
              LOG_WARN("Failed to find processes after %d attempts, continuing anyway", max_empty_retries);
              // Continue anyway, don't exit the loop
              pidfound = 1; // Force continue to avoid getting stuck
            }
            continue;
          }
        }

        // Reset counter when we find processes
        consecutive_empty_processes = 0;

        init_gpu_device_utilization();
        get_used_gpu_utilization(userutil,&sysprocnum);
        //if (sysprocnum == 1 &&
        //    userutil < upper_limit / 10) {
        //    g_cur_cuda_cores =
        //        delta(upper_limit, userutil, share);
        //    continue;
        //}

        // Check if we need to increase total CUDA cores to avoid negative values
        if ((share==g_total_cuda_cores) && (g_cur_cuda_cores<0)) {
          g_total_cuda_cores *= 2;
          share = g_total_cuda_cores;
        }

        // Only update if utilization values are in valid range
        if ((userutil[0]<=100) && (userutil[0]>=0)){
          share = delta(upper_limit, userutil[0], share);
          change_token(share);
        } else {
          // Handle invalid utilization values gracefully
          LOG_WARN("Invalid utilization value detected: %d, skipping update", userutil[0]);
        }
        LOG_INFO("userutil1=%d currentcores=%ld total=%ld limit=%d share=%ld\n",userutil[0],g_cur_cuda_cores,g_total_cuda_cores,upper_limit,share);
    }
}

void init_utilization_watcher() {
    LOG_INFO("set core utilization limit to  %d",get_current_device_sm_limit(0));

    // Initialize CUDA to NVML device mapping
    memset(cuda_to_nvml_map_array, 0, sizeof(cuda_to_nvml_map_array));
    unsigned int nvmlCounts;
    if (nvmlDeviceGetCount(&nvmlCounts) == NVML_SUCCESS) {
        for (unsigned int i = 0; i < nvmlCounts && i < CUDA_DEVICE_MAX_COUNT; i++) {
            cuda_to_nvml_map_array[i] = i;  // Default 1:1 mapping
        }
    }

    setspec();

    // Clean up any dead processes before starting the watcher
    rm_quitted_process();

    pthread_t tid;
    if ((get_current_device_sm_limit(0)<=100) && (get_current_device_sm_limit(0)>0)){
        pthread_create(&tid, NULL, utilization_watcher, NULL);
    }
    return;
}


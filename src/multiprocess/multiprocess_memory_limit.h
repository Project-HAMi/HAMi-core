#ifndef __MULTIPROCESS_MEMORY_LIMIT_H__
#define __MULTIPROCESS_MEMORY_LIMIT_H__

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <pthread.h>
#include <stdatomic.h>

#include "static_config.h"
#include "include/log_utils.h"


#define MULTIPROCESS_SHARED_REGION_MAGIC_FLAG  19920718
#define INIT_STATE_UNINIT 0
#define INIT_STATE_IN_PROGRESS 1
#define INIT_STATE_COMPLETE MULTIPROCESS_SHARED_REGION_MAGIC_FLAG
#define MULTIPROCESS_SHARED_REGION_CACHE_ENV   "CUDA_DEVICE_MEMORY_SHARED_CACHE"
#define MULTIPROCESS_SHARED_REGION_CACHE_DEFAULT  "/tmp/cudevshr.cache"
#define ENV_OVERRIDE_FILE "/overrideEnv"
#define CUDA_TASK_PRIORITY_ENV "CUDA_TASK_PRIORITY"

#define CUDA_DEVICE_MAX_COUNT 16
#define CUDA_DEVICE_MEMORY_UPDATE_SUCCESS 0
#define CUDA_DEVICE_MEMORY_UPDATE_FAILURE 1
#define MEMORY_LIMIT_TOLERATION_RATE 1.1

#define SHARED_REGION_SIZE_MAGIC  sizeof(shared_region_t)
#define SHARED_REGION_MAX_PROCESS_NUM 1024

// macros for debugging
#define SEQ_FIX_SHRREG_ACQUIRE_FLOCK_OK 0
#define SEQ_FIX_SHRREG_UPDATE_OWNER_OK 1
#define SEQ_FIX_SHRREG_RELEASE_FLOCK_OK 2
#define SEQ_ACQUIRE_SEMLOCK_OK 3
#define SEQ_UPDATE_OWNER_OK 4
#define SEQ_RESET_OWNER_OK 5
#define SEQ_RELEASE_SEMLOCK_OK 6
#define SEQ_BEFORE_UNLOCK_SHRREG 7

#define SEQ_AFTER_INC 8
#define SEQ_AFTER_DEC 9

#ifndef SEQ_POINT_MARK
    #define SEQ_POINT_MARK(s) 
#endif

#define FACTOR 32

#define MAJOR_VERSION 1
#define MINOR_VERSION 1

typedef struct {
    _Atomic uint64_t context_size;
    _Atomic uint64_t module_size;
    _Atomic uint64_t data_size;
    _Atomic uint64_t offset;
    _Atomic uint64_t total;
    uint64_t unused[3];
} device_memory_t;

typedef struct {
    _Atomic uint64_t dec_util;
    _Atomic uint64_t enc_util;
    _Atomic uint64_t sm_util;
    uint64_t unused[3];
} device_util_t;

typedef struct {
    _Atomic int32_t pid;           // Atomic to detect slot allocation
    _Atomic int32_t hostpid;
    _Atomic uint64_t seqlock;      // Sequence lock for consistent snapshots
    device_memory_t used[CUDA_DEVICE_MAX_COUNT];
    _Atomic uint64_t monitorused[CUDA_DEVICE_MAX_COUNT];
    device_util_t device_util[CUDA_DEVICE_MAX_COUNT];
    _Atomic int32_t status;
    uint64_t unused[2];
} shrreg_proc_slot_t;

typedef char uuid[96];

typedef struct {
    _Atomic int32_t initialized_flag;
    uint32_t major_version;
    uint32_t minor_version;
    _Atomic int32_t sm_init_flag;
    _Atomic size_t owner_pid;
    sem_t sem;  // Only for process slot add/remove
    sem_t sem_postinit;  // For serializing postInit() host PID detection
    uint64_t device_num;
    uuid uuids[CUDA_DEVICE_MAX_COUNT];
    uint64_t limit[CUDA_DEVICE_MAX_COUNT];
    uint64_t sm_limit[CUDA_DEVICE_MAX_COUNT];
    shrreg_proc_slot_t procs[SHARED_REGION_MAX_PROCESS_NUM];
    _Atomic int proc_num;
    _Atomic int utilization_switch;
    _Atomic int recent_kernel;
    int priority;
    _Atomic uint64_t last_kernel_time;
    uint64_t unused[4];
} shared_region_t;

typedef struct {
    int32_t pid;
    int fd;
    pthread_once_t init_status;
    shared_region_t* shared_region;
    uint64_t last_kernel_time; // cache for current process
    shrreg_proc_slot_t* my_slot;  // Cached pointer to this process's slot (lock-free access)
} shared_region_info_t;


typedef struct {
  size_t tid;
  CUcontext ctx;
} thread_context_map;

void ensure_initialized();

int get_current_device_sm_limit(int dev);
uint64_t get_current_device_memory_limit(const int dev);
int set_current_device_memory_limit(const int dev,size_t newlimit);
int set_current_device_sm_limit(int dev,int scale);
int set_current_device_sm_limit_scale(int dev,int scale);
int update_host_pid();
int set_host_pid(int hostpid);

uint64_t get_current_device_memory_monitor(const int dev);
uint64_t get_current_device_memory_usage(const int dev);
size_t get_gpu_memory_usage(const int dev);

// Priority-related
int get_current_priority();
int set_recent_kernel(int value);
int get_recent_kernel();
int get_utilization_switch();
int set_env_utilization_switch();

int set_gpu_device_memory_monitor(int32_t pid,int dev,size_t monitor);
int set_gpu_device_sm_utilization(int32_t pid,int dev, unsigned int smUtil);
int init_gpu_device_utilization();
int add_gpu_device_memory_usage(int32_t pid,int dev,size_t usage,int type);
int rm_gpu_device_memory_usage(int32_t pid,int dev,size_t usage,int type);

shrreg_proc_slot_t *find_proc_by_hostpid(int hostpid);
int active_oom_killer();
void pre_launch_kernel();

int shrreg_major_version();
int shrreg_minor_version();
int init_device_info();

//void inc_current_device_memory_usage(const int dev, const uint64_t usage);
//void decl_current_device_memory_usage(const int dev, const uint64_t usage);

//int oom_check(const int dev,int addon);

void lock_shrreg();
void unlock_shrreg();

int lock_postinit();  // Returns 1 on success, 0 on timeout
void unlock_postinit();

//Setspec of the corresponding device
int setspec();
//Remove quit process

void suspend_all();
void resume_all();
int wait_status_self(int status);
int wait_status_all(int status);
void print_all();

int load_env_from_file(char *filename);
int comparelwr(const char *s1,char *s2);
int put_device_info();
unsigned int nvml_to_cuda_map(unsigned int nvmldev);
unsigned int cuda_to_nvml_map(unsigned int cudadev);

int clear_proc_slot_nolock(int);
#endif  // __MULTIPROCESS_MEMORY_LIMIT_H__


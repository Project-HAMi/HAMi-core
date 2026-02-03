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
#include <stdatomic.h>

#include <assert.h>
#include <cuda.h>
#include "include/nvml_prefix.h"
#include <nvml.h>

#include "include/process_utils.h"
#include "include/memory_limit.h"
#include "multiprocess/multiprocess_memory_limit.h"


#ifndef SEM_WAIT_TIME
#define SEM_WAIT_TIME 10
#endif

#ifndef SEM_WAIT_TIME_ON_EXIT
#define SEM_WAIT_TIME_ON_EXIT 3
#endif

#ifndef SEM_WAIT_RETRY_TIMES
#define SEM_WAIT_RETRY_TIMES 30
#endif

// Longer timeout for postinit since set_task_pid() with adaptive polling can take several seconds
#ifndef SEM_WAIT_TIME_POSTINIT
#define SEM_WAIT_TIME_POSTINIT 30
#endif

#ifndef SEM_WAIT_RETRY_TIMES_POSTINIT
#define SEM_WAIT_RETRY_TIMES_POSTINIT 10
#endif

int pidfound;

int ctx_activate[32];

static shared_region_info_t region_info = {0, -1, PTHREAD_ONCE_INIT, NULL, 0, NULL};
//size_t initial_offset=117440512;
int env_utilization_switch;
int enable_active_oom_killer;
size_t context_size;
size_t initial_offset=0;
//lock for record kernel time
pthread_mutex_t _kernel_mutex;
int _record_kernel_interval = 1;

// forwards

void do_init_device_memory_limits(uint64_t*, int);
void exit_withlock(int exitcode);

void set_current_gpu_status(int status){
    // Fast path: use cached slot if available
    if (region_info.my_slot != NULL) {
        atomic_store_explicit(&region_info.my_slot->status, status, memory_order_release);
        return;
    }

    // Slow path: search for our slot
    int proc_num = atomic_load_explicit(&region_info.shared_region->proc_num, memory_order_acquire);
    int i;
    int32_t my_pid = getpid();
    for (i=0;i<proc_num;i++) {
        int32_t slot_pid = atomic_load_explicit(&region_info.shared_region->procs[i].pid, memory_order_acquire);
        if (my_pid == slot_pid){
            atomic_store_explicit(&region_info.shared_region->procs[i].status, status, memory_order_release);
            return;
        }
    }
}

void sig_restore_stub(int signo){
    set_current_gpu_status(1);
}

void sig_swap_stub(int signo){
    set_current_gpu_status(2);
}


// get device memory from env
size_t get_limit_from_env(const char* env_name) {
    char* env_limit = getenv(env_name);
    if (env_limit == NULL) {
        // fprintf(stderr, "No %s set in environment\n", env_name);
        return 0;
    }
    size_t len = strlen(env_limit);
    if (len == 0) {
        // fprintf(stderr, "Empty %s set in environment\n", env_name);
        return 0;
    }
    size_t scalar = 1;
    char* digit_end = env_limit + len;
    if (env_limit[len - 1] == 'G' || env_limit[len - 1] == 'g') {
        digit_end -= 1;
        scalar = 1024 * 1024 * 1024;
    } else if (env_limit[len - 1] == 'M' || env_limit[len - 1] == 'm') {
        digit_end -= 1;
        scalar = 1024 * 1024;
    } else if (env_limit[len - 1] == 'K' || env_limit[len - 1] == 'k') {
        digit_end -= 1;
        scalar = 1024;
    }
    size_t res = strtoul(env_limit, &digit_end, 0);
    size_t scaled_res = res * scalar;
    if (scaled_res == 0) {
        if (env_name[12]=='S'){
            LOG_INFO("device core util limit set to 0, which means no limit: %s=%s",
                env_name, env_limit);
        }else if (env_name[12]=='M'){
            LOG_WARN("invalid device memory limit %s=%s",env_name,env_limit);
        }else{
            LOG_WARN("invalid env name:%s",env_name);
        }
        return 0;
    }
    if (scaled_res != 0 && scaled_res / scalar != res) {
        LOG_ERROR("Limit overflow: %s=%s\n", env_name, env_limit);
        return 0;
    }
    return scaled_res;
}

int init_device_info() {
    unsigned int i,nvmlDevicesCount;
    CHECK_NVML_API(nvmlDeviceGetCount_v2(&nvmlDevicesCount));
    region_info.shared_region->device_num=nvmlDevicesCount;
    nvmlDevice_t dev;
    for(i=0;i<nvmlDevicesCount;i++){
        CHECK_NVML_API(nvmlDeviceGetHandleByIndex(i, &dev));
        CHECK_NVML_API(nvmlDeviceGetUUID(dev,region_info.shared_region->uuids[i],NVML_DEVICE_UUID_V2_BUFFER_SIZE));
    }
    LOG_INFO("put_device_info finished %d",nvmlDevicesCount);
    return 0;
}


int load_env_from_file(char *filename) {
    FILE *f=fopen(filename,"r");
    if (f==NULL)
        return 0;
    char tmp[10000];
    int cursor=0;
    while (!feof(f)){
        fgets(tmp,10000,f);
        if (strstr(tmp,"=")==NULL)
            break;
        if (tmp[strlen(tmp)-1]=='\n')
            tmp[strlen(tmp)-1]='\0';
        for (cursor=0;cursor<strlen(tmp);cursor++){
            if (tmp[cursor]=='=') {
                tmp[cursor]='\0';
                setenv(tmp,tmp+cursor+1,1);
                LOG_INFO("SET %s to %s",tmp,tmp+cursor+1);
                break;
            }
        }
    }
    return 0;
}

void do_init_device_memory_limits(uint64_t* arr, int len) {
    size_t fallback_limit = get_limit_from_env(CUDA_DEVICE_MEMORY_LIMIT);
    int i;
    for (i = 0; i < len; ++i) {
        char env_name[CUDA_DEVICE_MEMORY_LIMIT_KEY_LENGTH] = CUDA_DEVICE_MEMORY_LIMIT;
        char index_name[8];
        snprintf(index_name, 8, "_%d", i);
        strcat(env_name, index_name);
        size_t cur_limit = get_limit_from_env(env_name);
        if (cur_limit > 0) {
            arr[i] = cur_limit;
        } else if (fallback_limit > 0) {
            arr[i] = fallback_limit;
        } else {
            arr[i] = 0;
        }
    }
}

void do_init_device_sm_limits(uint64_t *arr, int len) {
    size_t fallback_limit = get_limit_from_env(CUDA_DEVICE_SM_LIMIT);
    if (fallback_limit == 0) fallback_limit = 100;
    int i;
    for (i = 0; i < len; ++i) {
        char env_name[CUDA_DEVICE_SM_LIMIT_KEY_LENGTH] = CUDA_DEVICE_SM_LIMIT;
        char index_name[8];
        snprintf(index_name, 8, "_%d", i);
        strcat(env_name, index_name);
        size_t cur_limit = get_limit_from_env(env_name);
        if (cur_limit > 0) {
            arr[i] = cur_limit;
        } else if (fallback_limit > 0) {
            arr[i] = fallback_limit;
        } else {
            arr[i] = 0;
        }
    }
}

int active_oom_killer() {
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        kill(region_info.shared_region->procs[i].pid,9);
    }
    return 0;
}

void pre_launch_kernel() {
    uint64_t now = time(NULL);
    pthread_mutex_lock(&_kernel_mutex);
    if (now - region_info.last_kernel_time < _record_kernel_interval) {
        pthread_mutex_unlock(&_kernel_mutex);
        return;
    }
    region_info.last_kernel_time = now;
    pthread_mutex_unlock(&_kernel_mutex);
    LOG_INFO("write last kernel time: %ld", now)
    lock_shrreg();
    if (region_info.shared_region->last_kernel_time < now) {
        region_info.shared_region->last_kernel_time = now;
    }
    unlock_shrreg();
}

int shrreg_major_version() {
    return MAJOR_VERSION;
}

int shrreg_minor_version() {
    return MINOR_VERSION;
}


size_t get_gpu_memory_monitor(const int dev) {
    LOG_DEBUG("get_gpu_memory_monitor dev=%d",dev);
    ensure_initialized();
    int i=0;
    size_t total=0;
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        LOG_DEBUG("dev=%d i=%lu,%lu\n",dev,region_info.shared_region->procs[i].monitorused[dev],region_info.shared_region->procs[i].used[dev].total);
        total+=region_info.shared_region->procs[i].monitorused[dev];
    }
    unlock_shrreg();
    return total;
}

// Lock-free memory usage aggregation with seqlock for consistent snapshots
size_t get_gpu_memory_usage(const int dev) {
    LOG_INFO("get_gpu_memory_usage_lockfree dev=%d",dev);
    ensure_initialized();
    int i=0;
    size_t total=0;

    // Lock-free read with acquire semantics for proc_num
    int proc_num = atomic_load_explicit(&region_info.shared_region->proc_num, memory_order_acquire);

    for (i=0;i<proc_num;i++){
        shrreg_proc_slot_t* slot = &region_info.shared_region->procs[i];
        uint64_t proc_usage;
        uint64_t seq1, seq2;
        int retry_count = 0;
        const int MAX_RETRIES = 100;

        // Seqlock read protocol: retry until we get a consistent snapshot
        do {
            // Read sequence number (must be even = no write in progress)
            seq1 = atomic_load_explicit(&slot->seqlock, memory_order_acquire);

            // If odd, writer is in progress, spin briefly
            while (seq1 & 1) {
                // CPU pause instruction to avoid hammering cache
                #if defined(__x86_64__) || defined(__i386__)
                __asm__ __volatile__("pause" ::: "memory");
                #elif defined(__aarch64__)
                __asm__ __volatile__("yield" ::: "memory");
                #endif
                seq1 = atomic_load_explicit(&slot->seqlock, memory_order_acquire);

                if (++retry_count > MAX_RETRIES) {
                    LOG_WARN("Seqlock retry limit exceeded for slot %d, using best-effort read", i);
                    goto best_effort_read;
                }
            }

            // Read the data with acquire semantics
            proc_usage = atomic_load_explicit(&slot->used[dev].total, memory_order_acquire);

            // Memory barrier to prevent reordering
            atomic_thread_fence(memory_order_acquire);

            // Read sequence number again
            seq2 = atomic_load_explicit(&slot->seqlock, memory_order_acquire);

            // If sequence numbers match and still even, read was consistent
        } while (seq1 != seq2);

        // Consistent read obtained
        int32_t pid = atomic_load_explicit(&slot->pid, memory_order_relaxed);
        int32_t hostpid = atomic_load_explicit(&slot->hostpid, memory_order_relaxed);

        LOG_INFO("dev=%d pid=%d host pid=%d i=%lu",dev,pid,hostpid,proc_usage);
        total+=proc_usage;
        continue;

best_effort_read:
        // Fallback: best-effort read if spinning too long
        proc_usage = atomic_load_explicit(&slot->used[dev].total, memory_order_acquire);
        pid = atomic_load_explicit(&slot->pid, memory_order_relaxed);
        hostpid = atomic_load_explicit(&slot->hostpid, memory_order_relaxed);
        LOG_WARN("dev=%d pid=%d host pid=%d i=%lu (best-effort)",dev,pid,hostpid,proc_usage);
        total+=proc_usage;
    }

    total+=initial_offset;
    return total;
}

int set_gpu_device_memory_monitor(int32_t pid,int dev,size_t monitor){
    //LOG_WARN("set_gpu_device_memory_monitor:%d %d %lu",pid,dev,monitor);
    int i;
    ensure_initialized();
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].hostpid == pid){
            LOG_INFO("set_gpu_device_memory_monitor:%d %d %lu->%lu",pid,dev,region_info.shared_region->procs[i].used[dev].total,monitor);
            region_info.shared_region->procs[i].monitorused[dev] = monitor;
            break;
        }
    }
    unlock_shrreg();
    return 1;
}

// Lock-free SM utilization update
int set_gpu_device_sm_utilization(int32_t pid,int dev, unsigned int smUtil){
    int i;
    ensure_initialized();

    int proc_num = atomic_load_explicit(&region_info.shared_region->proc_num, memory_order_acquire);
    for (i=0;i<proc_num;i++){
        int32_t hostpid = atomic_load_explicit(&region_info.shared_region->procs[i].hostpid, memory_order_acquire);
        if (hostpid == pid){
            uint64_t old_util = atomic_load_explicit(&region_info.shared_region->procs[i].device_util[dev].sm_util, memory_order_relaxed);
            LOG_INFO("set_gpu_device_sm_utilization_lockfree:%d %d %lu->%u", pid, dev, old_util, smUtil);
            atomic_store_explicit(&region_info.shared_region->procs[i].device_util[dev].sm_util, smUtil, memory_order_relaxed);
            return 1;
        }
    }
    return 0;
}

int init_gpu_device_utilization(){
    int i,dev;
    ensure_initialized();
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        for (dev=0;dev<CUDA_DEVICE_MAX_COUNT;dev++){
            region_info.shared_region->procs[i].device_util[dev].sm_util = 0;
            region_info.shared_region->procs[i].monitorused[dev] = 0;
            break;
        }
    }
    unlock_shrreg();
    return 1;
}

uint64_t nvml_get_device_memory_usage(const int dev) {
    nvmlDevice_t ndev;
    nvmlReturn_t ret;
    ret = nvmlDeviceGetHandleByIndex(dev, &ndev);
    if (ret != NVML_SUCCESS) {
        LOG_ERROR("NVML get device %d error, %s", dev, nvmlErrorString(ret));
    }
    unsigned int pcnt = SHARED_REGION_MAX_PROCESS_NUM;
    nvmlProcessInfo_v1_t infos[SHARED_REGION_MAX_PROCESS_NUM];
    LOG_DEBUG("before nvmlDeviceGetComputeRunningProcesses");
    ret = nvmlDeviceGetComputeRunningProcesses(ndev, &pcnt, infos);
    if (ret != NVML_SUCCESS) {
        LOG_ERROR("NVML get process error, %s", nvmlErrorString(ret));
    }
    int i = 0;
    uint64_t usage = 0;
    shared_region_t* region = region_info.shared_region;
    lock_shrreg();
    for (; i < pcnt; i++) {
        int slot = 0;
        for (; slot < region->proc_num; slot++) {
            if (infos[i].pid != region->procs[slot].pid)
                continue;
            usage += infos[i].usedGpuMemory;
        }
    }
    unlock_shrreg();
    LOG_DEBUG("Device %d current memory %lu / %lu", 
            dev, usage, region->limit[dev]);
    return usage;
}

// Lock-free memory add using atomics with seqlock for consistent reads
int add_gpu_device_memory_usage(int32_t pid,int cudadev,size_t usage,int type){
    LOG_INFO("add_gpu_device_memory_lockfree:%d %d->%d %lu",pid,cudadev,cuda_to_nvml_map(cudadev),usage);
    int dev = cuda_to_nvml_map(cudadev);
    ensure_initialized();

    // Fast path: use cached slot pointer for our own process
    if (pid == getpid() && region_info.my_slot != NULL) {
        shrreg_proc_slot_t* slot = region_info.my_slot;

        // Seqlock protocol: increment to odd (write in progress)
        atomic_fetch_add_explicit(&slot->seqlock, 1, memory_order_release);

        // Perform updates with release semantics for visibility
        atomic_fetch_add_explicit(&slot->used[dev].total, usage, memory_order_release);
        switch (type) {
            case 0:
                atomic_fetch_add_explicit(&slot->used[dev].context_size, usage, memory_order_release);
                break;
            case 1:
                atomic_fetch_add_explicit(&slot->used[dev].module_size, usage, memory_order_release);
                break;
            case 2:
                atomic_fetch_add_explicit(&slot->used[dev].data_size, usage, memory_order_release);
                break;
        }

        // Seqlock protocol: increment to even (write complete)
        atomic_fetch_add_explicit(&slot->seqlock, 1, memory_order_release);

        LOG_INFO("gpu_device_memory_added_lockfree:%d %d %lu",pid,dev,usage);
        return 0;
    }

    // Slow path: find slot for other process (still lock-free)
    int proc_num = atomic_load_explicit(&region_info.shared_region->proc_num, memory_order_acquire);
    int i;
    for (i=0;i<proc_num;i++){
        int32_t slot_pid = atomic_load_explicit(&region_info.shared_region->procs[i].pid, memory_order_acquire);
        if (slot_pid == pid){
            shrreg_proc_slot_t* slot = &region_info.shared_region->procs[i];

            // Seqlock protocol: increment to odd (write in progress)
            atomic_fetch_add_explicit(&slot->seqlock, 1, memory_order_release);

            // Perform updates
            atomic_fetch_add_explicit(&slot->used[dev].total, usage, memory_order_release);
            switch (type) {
                case 0:
                    atomic_fetch_add_explicit(&slot->used[dev].context_size, usage, memory_order_release);
                    break;
                case 1:
                    atomic_fetch_add_explicit(&slot->used[dev].module_size, usage, memory_order_release);
                    break;
                case 2:
                    atomic_fetch_add_explicit(&slot->used[dev].data_size, usage, memory_order_release);
                    break;
            }

            // Seqlock protocol: increment to even (write complete)
            atomic_fetch_add_explicit(&slot->seqlock, 1, memory_order_release);

            LOG_INFO("gpu_device_memory_added_lockfree:%d %d %lu",pid,dev,usage);
            return 0;
        }
    }

    LOG_WARN("Process slot not found for pid %d", pid);
    return -1;
}

// Lock-free memory remove using atomics with seqlock for consistent reads
int rm_gpu_device_memory_usage(int32_t pid,int cudadev,size_t usage,int type){
    LOG_INFO("rm_gpu_device_memory_lockfree:%d %d->%d %d:%lu",pid,cudadev,cuda_to_nvml_map(cudadev),type,usage);
    int dev = cuda_to_nvml_map(cudadev);
    ensure_initialized();

    // Fast path: use cached slot pointer for our own process
    if (pid == getpid() && region_info.my_slot != NULL) {
        shrreg_proc_slot_t* slot = region_info.my_slot;

        // Seqlock protocol: increment to odd (write in progress)
        atomic_fetch_add_explicit(&slot->seqlock, 1, memory_order_release);

        // Perform updates with release semantics
        atomic_fetch_sub_explicit(&slot->used[dev].total, usage, memory_order_release);
        switch (type) {
            case 0:
                atomic_fetch_sub_explicit(&slot->used[dev].context_size, usage, memory_order_release);
                break;
            case 1:
                atomic_fetch_sub_explicit(&slot->used[dev].module_size, usage, memory_order_release);
                break;
            case 2:
                atomic_fetch_sub_explicit(&slot->used[dev].data_size, usage, memory_order_release);
                break;
        }

        // Seqlock protocol: increment to even (write complete)
        atomic_fetch_add_explicit(&slot->seqlock, 1, memory_order_release);

        uint64_t new_total = atomic_load_explicit(&slot->used[dev].total, memory_order_acquire);
        LOG_INFO("after delete_lockfree:%lu",new_total);
        return 0;
    }

    // Slow path: find slot for other process (still lock-free)
    int proc_num = atomic_load_explicit(&region_info.shared_region->proc_num, memory_order_acquire);
    int i;
    for (i=0;i<proc_num;i++){
        int32_t slot_pid = atomic_load_explicit(&region_info.shared_region->procs[i].pid, memory_order_acquire);
        if (slot_pid == pid){
            shrreg_proc_slot_t* slot = &region_info.shared_region->procs[i];

            // Seqlock protocol: increment to odd (write in progress)
            atomic_fetch_add_explicit(&slot->seqlock, 1, memory_order_release);

            // Perform updates
            atomic_fetch_sub_explicit(&slot->used[dev].total, usage, memory_order_release);
            switch (type) {
                case 0:
                    atomic_fetch_sub_explicit(&slot->used[dev].context_size, usage, memory_order_release);
                    break;
                case 1:
                    atomic_fetch_sub_explicit(&slot->used[dev].module_size, usage, memory_order_release);
                    break;
                case 2:
                    atomic_fetch_sub_explicit(&slot->used[dev].data_size, usage, memory_order_release);
                    break;
            }

            // Seqlock protocol: increment to even (write complete)
            atomic_fetch_add_explicit(&slot->seqlock, 1, memory_order_release);

            uint64_t new_total = atomic_load_explicit(&slot->used[dev].total, memory_order_acquire);
            LOG_INFO("after delete_lockfree:%lu",new_total);
            return 0;
        }
    }

    LOG_WARN("Process slot not found for pid %d", pid);
    return -1;
}

void get_timespec(int seconds, struct timespec* spec) {
    struct timeval tv;
    gettimeofday(&tv, NULL);  // struggle with clock_gettime version
    spec->tv_sec = tv.tv_sec + seconds;
    spec->tv_nsec = 0;
}

int fix_lock_shrreg() {
    int res = 1;
    if (region_info.fd == -1) {
        // should never happen
        LOG_ERROR("Uninitialized shrreg");
    }
    // upgrade
    if (lockf(region_info.fd, F_LOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to upgraded lock: errno=%d", errno);
    }
    SEQ_POINT_MARK(SEQ_FIX_SHRREG_ACQUIRE_FLOCK_OK);

    shared_region_t* region = region_info.shared_region;
    int32_t current_owner = region->owner_pid;
    if (current_owner != 0) {
        int flag = 0;
        if (current_owner == region_info.pid) {
            LOG_INFO("Detect owner pid = self pid (%d), "
                "indicates pid loopback or race condition", current_owner);
            flag = 1;
        } else {
            int proc_status = proc_alive(current_owner);
            if (proc_status == PROC_STATE_NONALIVE) {
                LOG_INFO("Kick dead owner proc (%d)", current_owner);
                flag = 1;
            }
        }
        if (flag == 1) {
            LOG_INFO("Take upgraded lock (%d)", region_info.pid);
            region->owner_pid = region_info.pid;
            SEQ_POINT_MARK(SEQ_FIX_SHRREG_UPDATE_OWNER_OK);
            res = 0;     
        }
    }

    if (lockf(region_info.fd, F_ULOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to upgraded unlock: errno=%d", errno);
    }
    SEQ_POINT_MARK(SEQ_FIX_SHRREG_RELEASE_FLOCK_OK);
    return res;
}

void exit_withlock(int exitcode) {
    unlock_shrreg();
    exit(exitcode);
}


void exit_handler() {
    if (region_info.init_status == PTHREAD_ONCE_INIT) {
        return;
    }
    shared_region_t* region = region_info.shared_region;
    int slot = 0;
    LOG_MSG("Calling exit handler %d",getpid());
    struct timespec sem_ts;
    get_timespec(SEM_WAIT_TIME_ON_EXIT, &sem_ts);
    int status = sem_timedwait(&region->sem, &sem_ts);
    if (status == 0) {  // just give up on lock failure
        region->owner_pid = region_info.pid;
        while (slot < region->proc_num) {
            if (region->procs[slot].pid == region_info.pid) {
                memset(region->procs[slot].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
                memset(region->procs[slot].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
                region->proc_num--;
                region->procs[slot] = region->procs[region->proc_num];
                break;
            }
            slot++;
        }
        __sync_synchronize();
        region->owner_pid = 0;
        sem_post(&region->sem);
    } else {
        LOG_WARN("Failed to take lock on exit: errno=%d", errno);
    }
}


void lock_shrreg() {
    shared_region_t* region = region_info.shared_region;
    int trials = 0;
    while (1) {
        // CRITICAL: Create fresh timeout for each iteration!
        // If created outside loop, timestamp becomes stale after first timeout
        struct timespec sem_ts;
        get_timespec(SEM_WAIT_TIME, &sem_ts);

        int status = sem_timedwait(&region->sem, &sem_ts);
        SEQ_POINT_MARK(SEQ_ACQUIRE_SEMLOCK_OK);

        if (status == 0) {
            // TODO: irregular exit here will hang pending locks
            region->owner_pid = region_info.pid;
            __sync_synchronize();
            SEQ_POINT_MARK(SEQ_UPDATE_OWNER_OK);
            trials = 0;
            break;
        } else if (errno == ETIMEDOUT) {
            trials++;
            LOG_WARN("Lock shrreg timeout (trial %d/%d), try fix (%d:%ld)",
                     trials, SEM_WAIT_RETRY_TIMES, region_info.pid, region->owner_pid);
            int32_t current_owner = region->owner_pid;

            // Check if owner is dead or if this is our own PID (deadlock)
            if (current_owner != 0 && (current_owner == region_info.pid ||
                    proc_alive(current_owner) == PROC_STATE_NONALIVE)) {
                LOG_WARN("Owner proc dead or self-deadlock (%d), forcing recovery", current_owner);
                if (0 == fix_lock_shrreg()) {
                    break;  // Successfully recovered
                }
                // If fix failed, force-post the semaphore to unlock it
                LOG_WARN("fix_lock_shrreg failed, force-posting semaphore");
                sem_post(&region->sem);
                continue;
            }

            // If too many retries, force recovery even if owner seems alive
            if (trials > SEM_WAIT_RETRY_TIMES) {
                LOG_WARN("Exceeded retry limit (%d sec), forcing recovery",
                         SEM_WAIT_RETRY_TIMES * SEM_WAIT_TIME);
                if (current_owner == 0) {
                    LOG_WARN("Owner is 0, setting to %d", region_info.pid);
                    region->owner_pid = region_info.pid;
                }
                if (0 == fix_lock_shrreg()) {
                    break;
                }
                // Last resort: force-post semaphore
                LOG_WARN("All recovery attempts failed, force-posting semaphore");
                sem_post(&region->sem);
                continue;
            }
            continue;  // Retry with backoff
        } else {
            LOG_ERROR("Failed to lock shrreg: %d", errno);
        }
    }
}

void unlock_shrreg() {
    SEQ_POINT_MARK(SEQ_BEFORE_UNLOCK_SHRREG);
    shared_region_t* region = region_info.shared_region;

    __sync_synchronize();
    region->owner_pid = 0;
    // TODO: irregular exit here will hang pending locks
    SEQ_POINT_MARK(SEQ_RESET_OWNER_OK);

    sem_post(&region->sem);
    SEQ_POINT_MARK(SEQ_RELEASE_SEMLOCK_OK);
}

int lock_postinit() {
    shared_region_t* region = region_info.shared_region;
    int trials = 0;
    while (1) {
        // CRITICAL: Create fresh timeout for each iteration!
        // If created outside loop, timestamp becomes stale after first timeout
        // Use longer timeout for postinit since set_task_pid() can take several seconds
        struct timespec sem_ts;
        get_timespec(SEM_WAIT_TIME_POSTINIT, &sem_ts);

        int status = sem_timedwait(&region->sem_postinit, &sem_ts);
        if (status == 0) {
            // Lock acquired successfully
            LOG_DEBUG("Acquired postinit lock after %d waits (PID %d)", trials, getpid());
            return 1;  // Success
        } else if (errno == ETIMEDOUT) {
            trials++;
            LOG_MSG("Waiting for postinit lock (trial %d/%d, waited %ds, PID %d)",
                    trials, SEM_WAIT_RETRY_TIMES_POSTINIT, trials * SEM_WAIT_TIME_POSTINIT, getpid());

            // After many retries, give up
            if (trials > SEM_WAIT_RETRY_TIMES_POSTINIT) {
                LOG_ERROR("Postinit lock timeout after %d seconds - another process may have crashed",
                          SEM_WAIT_RETRY_TIMES_POSTINIT * SEM_WAIT_TIME_POSTINIT);
                LOG_ERROR("Skipping host PID detection for this process (will use container PID)");
                return 0;  // Timeout - didn't acquire lock
            }
            continue;
        } else {
            LOG_ERROR("Failed to lock postinit semaphore: errno=%d", errno);
            // Don't give up - keep retrying
            trials++;
            continue;
        }
    }
}

void unlock_postinit() {
    shared_region_t* region = region_info.shared_region;
    sem_post(&region->sem_postinit);
}


int clear_proc_slot_nolock(int do_clear) {
    int slot = 0;
    int res=0;
    shared_region_t* region = region_info.shared_region;
    while (slot < region->proc_num) {
        int32_t pid = region->procs[slot].pid;
        if (pid != 0) {
            if (do_clear > 0 && proc_alive(pid) == PROC_STATE_NONALIVE) {
                LOG_WARN("Kick dead proc %d", pid);
            } else {
                slot++;
                continue;
            }
            res=1;
            region->proc_num--;
            region->procs[slot] = region->procs[region->proc_num];
            __sync_synchronize();
        }
    }
    return res;
}

void init_proc_slot_withlock() {
    int32_t current_pid = getpid();
    lock_shrreg();  // Still need lock for modifying process slots
    shared_region_t* region = region_info.shared_region;

    int proc_num = atomic_load_explicit(&region->proc_num, memory_order_acquire);
    if (proc_num >= SHARED_REGION_MAX_PROCESS_NUM) {
        exit_withlock(-1);
    }
    signal(SIGUSR2,sig_swap_stub);
    signal(SIGUSR1,sig_restore_stub);

    // If, by any means a pid of itself is found in region->process, then it is probably caused by crashloop
    // we need to reset it.
    int i,found=0;
    for (i=0; i<proc_num; i++) {
        int32_t slot_pid = atomic_load_explicit(&region->procs[i].pid, memory_order_acquire);
        if (slot_pid == current_pid) {
            atomic_store_explicit(&region->procs[i].seqlock, 0, memory_order_relaxed);  // Reset seqlock
            atomic_store_explicit(&region->procs[i].status, 1, memory_order_release);

            // Zero out atomics
            for (int dev=0; dev<CUDA_DEVICE_MAX_COUNT; dev++) {
                atomic_store_explicit(&region->procs[i].used[dev].total, 0, memory_order_relaxed);
                atomic_store_explicit(&region->procs[i].used[dev].context_size, 0, memory_order_relaxed);
                atomic_store_explicit(&region->procs[i].used[dev].module_size, 0, memory_order_relaxed);
                atomic_store_explicit(&region->procs[i].used[dev].data_size, 0, memory_order_relaxed);
                atomic_store_explicit(&region->procs[i].device_util[dev].sm_util, 0, memory_order_relaxed);
                atomic_store_explicit(&region->procs[i].monitorused[dev], 0, memory_order_relaxed);
            }

            region_info.my_slot = &region->procs[i];  // Cache our slot pointer
            found = 1;
            break;
        }
    }

    if (!found) {
        // Initialize new slot with atomics
        atomic_store_explicit(&region->procs[proc_num].seqlock, 0, memory_order_relaxed);  // Start with even (no write)
        atomic_store_explicit(&region->procs[proc_num].pid, current_pid, memory_order_release);
        atomic_store_explicit(&region->procs[proc_num].hostpid, 0, memory_order_relaxed);
        atomic_store_explicit(&region->procs[proc_num].status, 1, memory_order_release);

        for (int dev=0; dev<CUDA_DEVICE_MAX_COUNT; dev++) {
            atomic_store_explicit(&region->procs[proc_num].used[dev].total, 0, memory_order_relaxed);
            atomic_store_explicit(&region->procs[proc_num].used[dev].context_size, 0, memory_order_relaxed);
            atomic_store_explicit(&region->procs[proc_num].used[dev].module_size, 0, memory_order_relaxed);
            atomic_store_explicit(&region->procs[proc_num].used[dev].data_size, 0, memory_order_relaxed);
            atomic_store_explicit(&region->procs[proc_num].device_util[dev].sm_util, 0, memory_order_relaxed);
            atomic_store_explicit(&region->procs[proc_num].monitorused[dev], 0, memory_order_relaxed);
        }

        region_info.my_slot = &region->procs[proc_num];  // Cache our slot pointer
        atomic_fetch_add_explicit(&region->proc_num, 1, memory_order_release);
    }

    clear_proc_slot_nolock(1);
    unlock_shrreg();
}

void print_all() {
    int i;
    LOG_INFO("Total process: %d",region_info.shared_region->proc_num);
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        for (int dev=0;dev<CUDA_DEVICE_MAX_COUNT;dev++){
            LOG_INFO("Process %d hostPid: %d, sm: %lu, memory: %lu, record: %lu",
                region_info.shared_region->procs[i].pid,
                region_info.shared_region->procs[i].hostpid, 
                region_info.shared_region->procs[i].device_util[dev].sm_util, 
                region_info.shared_region->procs[i].monitorused[dev], 
                region_info.shared_region->procs[i].used[dev].total);
        }
    }
}

void child_reinit_flag() {
    LOG_DEBUG("Detect child pid: %d -> %d", region_info.pid, getpid());   
    region_info.init_status = PTHREAD_ONCE_INIT;
}

int set_active_oom_killer() {
    char *oom_killer_env;
    oom_killer_env = getenv("ACTIVE_OOM_KILLER");
    if (oom_killer_env!=NULL){
        if (strcmp(oom_killer_env,"false") == 0)
            return 0;
        if (strcmp(oom_killer_env,"true") == 0)
            return 1;
        if (strcmp(oom_killer_env,"0")==0)
            return 0;
        if (strcmp(oom_killer_env,"1")==0)
            return 1;
    }
    return 1;
}

int set_env_utilization_switch() {
    char *utilization_env;
    utilization_env = getenv("GPU_CORE_UTILIZATION_POLICY");
    if (utilization_env!=NULL){
        if ((strcmp(utilization_env,"FORCE") ==0 ) || (strcmp(utilization_env,"force") ==0))
            return 1;
        if ((strcmp(utilization_env,"DISABLE") ==0 ) || (strcmp(utilization_env,"disable") ==0 ))
            return 2;
    }
    return 0;
}

void try_create_shrreg() {
    LOG_DEBUG("Try create shrreg")
    if (region_info.fd == -1) {
        // use .fd to indicate whether a reinit after fork happen
        // no need to register exit handler after fork
        if (0 != atexit(exit_handler)) {
            LOG_ERROR("Register exit handler failed: %d", errno);
        }
    }

    enable_active_oom_killer = set_active_oom_killer();
    env_utilization_switch = set_env_utilization_switch();
    pthread_atfork(NULL, NULL, child_reinit_flag);

    region_info.pid = getpid();
    region_info.fd = -1;
    region_info.last_kernel_time = time(NULL);

    umask(0);

    char* shr_reg_file = getenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV);
    if (shr_reg_file == NULL) {
        shr_reg_file = MULTIPROCESS_SHARED_REGION_CACHE_DEFAULT;
    }
    // Initialize NVML BEFORE!! open it
    //nvmlInit();

    /* If you need sm modification, do it here */
    /* ... set_sm_scale */

    int fd = open(shr_reg_file, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        LOG_ERROR("Fail to open shrreg %s: errno=%d", shr_reg_file, errno);
    }
    region_info.fd = fd;
    size_t offset = lseek(fd, SHARED_REGION_SIZE_MAGIC, SEEK_SET);
    if (offset != SHARED_REGION_SIZE_MAGIC) {
        LOG_ERROR("Fail to init shrreg %s: errno=%d", shr_reg_file, errno);
    }
    size_t check_bytes = write(fd, "\0", 1);
    if (check_bytes != 1) {
        LOG_ERROR("Fail to write shrreg %s: errno=%d", shr_reg_file, errno);
    }
    if (lseek(fd, 0, SEEK_SET) != 0) {
        LOG_ERROR("Fail to reseek shrreg %s: errno=%d", shr_reg_file, errno);
    }
    region_info.shared_region = (shared_region_t*) mmap(
        NULL, SHARED_REGION_SIZE_MAGIC,
        PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    shared_region_t* region = region_info.shared_region;
    if (region == NULL) {
        LOG_ERROR("Fail to map shrreg %s: errno=%d", shr_reg_file, errno);
    }

    // ============================================================================
    // OPTION C: Atomic Double-Checked Locking (Eliminates File Lock!)
    // ============================================================================
    // Fast path: Check if already initialized (no lock needed)
    int32_t init_flag = atomic_load_explicit(&region->initialized_flag, memory_order_acquire);
    if (init_flag == INIT_STATE_COMPLETE) {
        // Already initialized by another process! Skip to validation
        LOG_DEBUG("Shared region already initialized, skipping init (fast path)");
        goto validate_limits;
    }

    // Slow path: Try to become the initializer using atomic CAS
    int32_t expected = INIT_STATE_UNINIT;
    if (atomic_compare_exchange_strong_explicit(
            &region->initialized_flag,
            &expected,
            INIT_STATE_IN_PROGRESS,
            memory_order_acquire,
            memory_order_acquire)) {

        // ========================================================================
        // WE WON THE RACE! This process is the designated initializer
        // ========================================================================
        LOG_INFO("Process %d won initializer race, performing initialization", getpid());

        // Initialize semaphores FIRST (needed for process slot and postInit serialization)
        if (sem_init(&region->sem, 1, 1) != 0) {
            LOG_ERROR("Fail to init sem %s: errno=%d", shr_reg_file, errno);
        }
        if (sem_init(&region->sem_postinit, 1, 1) != 0) {
            LOG_ERROR("Fail to init sem_postinit %s: errno=%d", shr_reg_file, errno);
        }

        // Initialize version and limits for ALL 8 GPUs
        region->major_version = MAJOR_VERSION;
        region->minor_version = MINOR_VERSION;
        do_init_device_memory_limits(region->limit, CUDA_DEVICE_MAX_COUNT);
        do_init_device_sm_limits(region->sm_limit, CUDA_DEVICE_MAX_COUNT);

        // Initialize atomic fields
        atomic_store_explicit(&region->sm_init_flag, 0, memory_order_relaxed);
        atomic_store_explicit(&region->utilization_switch, 1, memory_order_relaxed);
        atomic_store_explicit(&region->recent_kernel, 2, memory_order_relaxed);
        atomic_store_explicit(&region->proc_num, 0, memory_order_relaxed);

        region->priority = 1;
        if (getenv(CUDA_TASK_PRIORITY_ENV) != NULL)
            region->priority = atoi(getenv(CUDA_TASK_PRIORITY_ENV));

        // Release barrier: ensure all writes are visible before marking complete
        atomic_thread_fence(memory_order_release);

        // Mark initialization complete (releases waiting processes)
        atomic_store_explicit(&region->initialized_flag, INIT_STATE_COMPLETE, memory_order_release);

        LOG_INFO("Initialization complete by process %d", getpid());
    } else {
        // ========================================================================
        // Another process is initializing or already initialized
        // ========================================================================
        LOG_DEBUG("Process %d waiting for initialization by another process...", getpid());

        // Spin-wait for initialization to complete (should be fast, ~2 seconds max)
        int spin_count = 0;
        while (1) {
            init_flag = atomic_load_explicit(&region->initialized_flag, memory_order_acquire);
            if (init_flag == INIT_STATE_COMPLETE) {
                LOG_DEBUG("Process %d detected initialization complete after %d spins",
                         getpid(), spin_count);
                break;
            }

            // Avoid busy-waiting: small sleep reduces CPU usage
            usleep(1000);  // 1ms
            spin_count++;

            if (spin_count > 10000) {  // 10 seconds timeout
                LOG_ERROR("Timeout waiting for initialization (current state: %d)", init_flag);
                break;
            }
        }
    }

validate_limits:
    // ============================================================================
    // Validation: All processes check their environment matches shared state
    // ============================================================================
    if (region->initialized_flag == INIT_STATE_COMPLETE) {
        if (region->major_version != MAJOR_VERSION || 
                region->minor_version != MINOR_VERSION) {
            LOG_ERROR("The current version number %d.%d"
                    " is different from the file's version number %d.%d",
                    MAJOR_VERSION, MINOR_VERSION,
                    region->major_version, region->minor_version);
        }
        uint64_t local_limits[CUDA_DEVICE_MAX_COUNT];
        do_init_device_memory_limits(local_limits, CUDA_DEVICE_MAX_COUNT);
        int i;
        for (i = 0; i < CUDA_DEVICE_MAX_COUNT; ++i) {
            if (local_limits[i] != region->limit[i]) {
                LOG_ERROR("Limit inconsistency detected for %dth device"
                    ", %lu expected, get %lu", 
                    i, local_limits[i], region->limit[i]);
            }
        }
        do_init_device_sm_limits(local_limits,CUDA_DEVICE_MAX_COUNT);
        for (i = 0; i < CUDA_DEVICE_MAX_COUNT; ++i) {
            if (local_limits[i] != region->sm_limit[i]) {
                LOG_INFO("SM limit inconsistency detected for %dth device"
                    ", %lu expected, get %lu", 
                    i, local_limits[i], region->sm_limit[i]);
            //    exit(1); 
            }
        }
    }
    region->last_kernel_time = region_info.last_kernel_time;

    // No lockf unlock needed - we used atomic CAS instead of file lock!
    LOG_DEBUG("shrreg created");
}

void initialized() {
    pthread_mutex_init(&_kernel_mutex, NULL);
    char* _record_kernel_interval_env = getenv("RECORD_KERNEL_INTERVAL");
    if (_record_kernel_interval_env) {
        _record_kernel_interval = atoi(_record_kernel_interval_env);
    }
    try_create_shrreg();
    init_proc_slot_withlock();
}

void ensure_initialized() {
    (void) pthread_once(&region_info.init_status, initialized);
}

int update_host_pid() {
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == getpid()){
            if (region_info.shared_region->procs[i].hostpid!=0)
                pidfound=1; 
        }
    }
    return 0;
}

int set_host_pid(int hostpid) {
    int i,j,found=0;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == getpid()){
            LOG_INFO("SET PID= %d",hostpid);
            found=1;
            region_info.shared_region->procs[i].hostpid = hostpid;
            for (j=0;j<CUDA_DEVICE_MAX_COUNT;j++)
                region_info.shared_region->procs[i].monitorused[j]=0;
        }
    }
    if (!found) {
        LOG_ERROR("HOST PID NOT FOUND. %d",hostpid);
        return -1;
    }
    setspec();
    return 0;
}

int set_current_device_sm_limit_scale(int dev, int scale) {
    ensure_initialized();
    if (region_info.shared_region->sm_init_flag==1) return 0;
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    LOG_INFO("dev %d new sm limit set mul by %d",dev,scale);
    region_info.shared_region->sm_limit[dev]=region_info.shared_region->sm_limit[dev]*scale;
    region_info.shared_region->sm_init_flag = 1;
    return 0;
}

int get_current_device_sm_limit(int dev) {
    ensure_initialized();
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    return region_info.shared_region->sm_limit[dev];
}

int set_current_device_memory_limit(const int dev,size_t newlimit) {
    ensure_initialized();
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    LOG_INFO("dev %d new limit set to %ld",dev,newlimit);
    region_info.shared_region->limit[dev]=newlimit;
    return 0; 
}

uint64_t get_current_device_memory_limit(const int dev) {
    ensure_initialized();
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    return region_info.shared_region->limit[dev];       
}

uint64_t get_current_device_memory_monitor(const int dev) {
    ensure_initialized();
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    uint64_t result = get_gpu_memory_monitor(dev);
//    result= nvml_get_device_memory_usage(dev);
    return result;
}

uint64_t get_current_device_memory_usage(const int dev) {
    clock_t start,finish;
    uint64_t result;
    start = clock();
    ensure_initialized();
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    result = get_gpu_memory_usage(dev);
//    result= nvml_get_device_memory_usage(dev);
    finish=clock();
    LOG_DEBUG("get_current_device_memory_usage:tick=%lu result=%lu\n",finish-start,result);
    return result;
}

int get_current_priority() {
    return region_info.shared_region->priority;
}

int get_recent_kernel(){
    return region_info.shared_region->recent_kernel;
}

int set_recent_kernel(int value){
    region_info.shared_region->recent_kernel=value;
    return 0;
}

int get_utilization_switch() {
    if (env_utilization_switch==1)
        return 1;
    if (env_utilization_switch==2)
        return 0;
    return region_info.shared_region->utilization_switch; 
}

void suspend_all(){
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        LOG_INFO("Sending USR2 to %d",region_info.shared_region->procs[i].pid);
        kill(region_info.shared_region->procs[i].pid,SIGUSR2);
    }
}

void resume_all(){
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        LOG_INFO("Sending USR1 to %d",region_info.shared_region->procs[i].pid);
        kill(region_info.shared_region->procs[i].pid,SIGUSR1);
    }
}

int wait_status_self(int status){
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid==getpid()){
            if (region_info.shared_region->procs[i].status==status)
                return 1;
            else
                return 0;
        }
    }
    return -1;
}

int wait_status_all(int status){
    int i;
    int released = 1;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        LOG_INFO("i=%d pid=%d status=%d",i,region_info.shared_region->procs[i].pid,region_info.shared_region->procs[i].status);
        if ((region_info.shared_region->procs[i].status!=status) && (region_info.shared_region->procs[i].pid!=getpid()))
            released = 0; 
    }
    LOG_INFO("Return released=%d",released);
    return released;
}

shrreg_proc_slot_t *find_proc_by_hostpid(int hostpid) {
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        if (region_info.shared_region->procs[i].hostpid == hostpid) 
            return &region_info.shared_region->procs[i];
    }
    return NULL;
}


int comparelwr(const char *s1,char *s2){
    if ((s1==NULL) || (s2==NULL))
        return 1;
    if (strlen(s1)!=strlen(s2)) {
        return 1;
    }
    int i;
    for (i=0;i<strlen(s1);i++)
        if (tolower(s1[i])!=tolower(s2[i])){
            return 1;
        }
    return 0;
}

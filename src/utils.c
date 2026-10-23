#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <sys/file.h>
#include <unistd.h>
#include "include/utils.h"
#include "include/log_utils.h"
#include "include/nvml_prefix.h"
#include <nvml.h>
#include "include/nvml_override.h"
#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
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

static nvmlReturn_t get_used_gpu_memory_by_pid(pid_t hostpid, int cudadev,
                                              unsigned long long *used) {
    nvmlProcessInfo_v1_t *processes;
    nvmlDevice_t device;
    nvmlReturn_t res;
    unsigned int count = SHARED_REGION_MAX_PROCESS_NUM;
    unsigned int i;

    res = nvmlDeviceGetHandleByIndex(cuda_to_nvml_map(cudadev), &device);
    if (res != NVML_SUCCESS) {
        return res;
    }
    processes = calloc(count, sizeof(*processes));
    if (processes == NULL) {
        return NVML_ERROR_MEMORY;
    }
    res = nvmlDeviceGetComputeRunningProcesses(device, &count, processes);
    if (res == NVML_SUCCESS) {
        // Newer drivers report the caller's own-namespace PID instead of the
        // host PID, so prefer the host PID and fall back to getpid().
        const nvmlProcessInfo_v1_t *self = NULL, *match = NULL;

        for (i = 0; i < count; i++) {
            if (processes[i].pid == (unsigned int)hostpid) {
                match = &processes[i];
                break;
            }
            if (self == NULL && processes[i].pid == (unsigned int)getpid()) {
                self = &processes[i];
            }
        }
        if (match == NULL) {
            match = self;
        }
        res = NVML_ERROR_NOT_FOUND;
        if (match != NULL &&
            match->usedGpuMemory != (unsigned long long)NVML_VALUE_NOT_AVAILABLE) {
            *used = match->usedGpuMemory;
            res = NVML_SUCCESS;
        }
    }
    free(processes);
    return res;
}

static pthread_mutex_t lazy_sizing_lock = PTHREAD_MUTEX_INITIALIZER;
static pid_t lazy_sizing_hostpid;

// The broker names this process's host PID without NVML discovery. The
// context is sized later, from the first primary context the application
// retains, so init creates no context of its own.
nvmlReturn_t set_task_pid_from_broker(void) {
    pid_t hostpid = 0;

    if (hostpid_broker_query_trusted(HOSTPID_BROKER_SOCKET_PATH,
                                    &hostpid) != 0) {
        LOG_WARN("Host PID broker unavailable: %s", strerror(errno));
        return NVML_ERROR_UNKNOWN;
    }
    LOG_INFO("hostPid=%d from broker", hostpid);
    if (set_host_pid(hostpid) != 0) {
        return NVML_ERROR_NOT_FOUND;
    }
    if (nvmlInit() != NVML_SUCCESS) {
        LOG_WARN("Primary context size unavailable");
        return NVML_SUCCESS;
    }
    // The first nvmlInit (including setspec in set_host_pid) resets the map.
    map_cuda_visible_devices();
    pthread_mutex_lock(&lazy_sizing_lock);
    lazy_sizing_hostpid = hostpid;
    pthread_mutex_unlock(&lazy_sizing_lock);
    return NVML_SUCCESS;
}

// Called after a successful primary context retain while context_size is
// unknown. The broker PID lets NVML report this process's own usage, so no
// lock and no extra context are needed.
void size_primary_context_lazily(int cudadev) {
    unsigned long long used = 0;
    nvmlReturn_t res;

    pthread_mutex_lock(&lazy_sizing_lock);
    if (lazy_sizing_hostpid > 0 && context_size == 0) {
        res = get_used_gpu_memory_by_pid(lazy_sizing_hostpid, cudadev, &used);
        if (res == NVML_SUCCESS) {
            LOG_INFO("Primary Context Size==%llu", used);
            context_size = used;
        } else {
            LOG_WARN("Primary context size unavailable: %d", res);
        }
    }
    pthread_mutex_unlock(&lazy_sizing_lock);
}

void reset_lazy_context_sizing(void) {
    lazy_sizing_lock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    lazy_sizing_hostpid = 0;
}

/*
 * Host PID discovery. A process inside a container cannot read its own host PID, so set_task_pid()
 * looks for it in NVML's process list for the device, which reports host PIDs.
 */
#ifndef HOSTPID_DETECT_MAX_ATTEMPTS
#define HOSTPID_DETECT_MAX_ATTEMPTS 8
#endif

static int pid_listed(const nvmlProcessInfo_t1 *list, unsigned int count, unsigned int pid) {
    unsigned int i;
    for (i = 0; i < count; i++) {
        if (list[i].pid == pid)
            return 1;
    }
    return 0;
}

static int pid_in(const unsigned int *pids, unsigned int count, unsigned int pid) {
    unsigned int i;
    for (i = 0; i < count; i++) {
        if (pids[i] == pid)
            return 1;
    }
    return 0;
}

// out must not overlap known.
unsigned int own_pid_candidates(const nvmlProcessInfo_t1 *before, unsigned int n_before,
                                const nvmlProcessInfo_t1 *during, unsigned int n_during,
                                const nvmlProcessInfo_t1 *after, unsigned int n_after,
                                const unsigned int *known, unsigned int n_known,
                                unsigned int *out, unsigned int max_out) {
    unsigned int i, n = 0, kept = 0;
    // With known set, only PIDs that matched every earlier probe count; never restart from scratch.
    for (i = 0; i < n_during; i++) {
        unsigned int pid = during[i].pid;
        if (pid_listed(during, i, pid))
            continue;
        if (pid_listed(before, n_before, pid) || pid_listed(after, n_after, pid))
            continue;
        if (n_known > 0 && !pid_in(known, n_known, pid))
            continue;
        if (n < max_out)
            out[kept++] = pid;
        n++;
    }
    return n;
}

// Containers often share small PIDs and start in the same second, so mix in the clock's nanoseconds.
unsigned int hostpid_retry_seed(const struct timespec *ts, unsigned int pid) {
    return (unsigned int)ts->tv_sec ^ (unsigned int)ts->tv_nsec ^ (pid << 16) ^ pid;
}

static nvmlReturn_t list_compute_processes(nvmlDevice_t device, nvmlProcessInfo_t1 *out, unsigned int *count) {
    nvmlReturn_t res;
    *count = SHARED_REGION_MAX_PROCESS_NUM;
    // nvmlProcessInfo_t1 has the layout of nvmlProcessInfo_v1_t
    res = nvmlDeviceGetComputeRunningProcesses(device, count, (nvmlProcessInfo_v1_t *)out);
    if (res != NVML_SUCCESS) {
        LOG_ERROR("nvmlDeviceGetComputeRunningProcesses failed %d (count %u)", res, *count);
        *count = 0;
    }
    return res;
}

static int nvml_probe(void *ctx, hostpid_probe_t *p) {
    nvmlDevice_t device = *(nvmlDevice_t *)ctx;
    nvmlReturn_t res;
    CUcontext pctx;

    res = list_compute_processes(device, p->before, &p->n_before);
    if (res != NVML_SUCCESS)
        return res;
    CHECK_CU_RESULT(cuDevicePrimaryCtxRetain(&pctx, 0));
    res = list_compute_processes(device, p->during, &p->n_during);
    CHECK_CU_RESULT(cuDevicePrimaryCtxRelease(0));
    if (res != NVML_SUCCESS)
        return res;
    return list_compute_processes(device, p->after, &p->n_after);
}

_Static_assert(HOSTPID_PROBE_MAX == SHARED_REGION_MAX_PROCESS_NUM, "probe buffers must match the shared region");

// A candidate is accepted only after it matched two probes in a row, so a neighbour that appears
// and exits inside one window cannot be taken even when our own PID is missing from the list.
int find_own_hostpid(hostpid_probe_fn probe, void *ctx, int max_attempts,
                     unsigned int *hostpid, uint64_t *used) {
    hostpid_probe_t p;
    unsigned int known[SHARED_REGION_MAX_PROCESS_NUM];
    unsigned int found[SHARED_REGION_MAX_PROCESS_NUM];
    unsigned int n_known = 0, i, seed;
    struct timespec now;
    int attempt;

    clock_gettime(CLOCK_MONOTONIC, &now);
    seed = hostpid_retry_seed(&now, (unsigned int)getpid());
    for (attempt = 1; attempt <= max_attempts; attempt++) {
        struct timespec t0, t1;
        int res;
        unsigned int n;
        int confirmed;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        res = probe(ctx, &p);
        if (res != NVML_SUCCESS)
            return res;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        n = own_pid_candidates(p.before, p.n_before, p.during, p.n_during, p.after, p.n_after,
                               known, n_known, found, SHARED_REGION_MAX_PROCESS_NUM);
        LOG_INFO("host pid probe %d: %u processes before, %u during, %u after, %u candidates",
                 attempt, p.n_before, p.n_during, p.n_after, n);
        confirmed = n_known > 0;
        if (n > 0) {
            n_known = n < SHARED_REGION_MAX_PROCESS_NUM ? n : SHARED_REGION_MAX_PROCESS_NUM;
            memcpy(known, found, n_known * sizeof(known[0]));
        } else {
            n_known = 0;
            LOG_WARN("host pid probe %d: no candidate; our PID may still be listed after the release", attempt);
        }
        if (n == 1 && confirmed) {
            *hostpid = known[0];
            *used = NVML_VALUE_NOT_AVAILABLE;
            for (i = 0; i < p.n_during; i++) {
                if (p.during[i].pid == *hostpid) {
                    *used = p.during[i].usedGpuMemory;
                    break;
                }
            }
            return NVML_SUCCESS;
        }
        if (n != 1)
            LOG_WARN("host pid detection attempt %d/%d: %u candidates, retrying", attempt, max_attempts, n);
        if (attempt < max_attempts) {
            int64_t probe_us = (int64_t)(t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000;
            // Random delay up to twice the probe time, to break lockstep between processes
            usleep(1000 + rand_r(&seed) % (unsigned int)(2 * probe_us + 1000));
        }
    }
    LOG_ERROR("host pid is error!");
    return NVML_ERROR_DRIVER_NOT_LOADED;
}

nvmlReturn_t set_task_pid() {
    unsigned int hostpid;
    uint64_t used;
    nvmlDevice_t device;
    nvmlReturn_t res;

    CHECK_NVML_API(nvmlInit());
    // nvml_probe retains the primary context of CUDA device 0, so probe that device's NVML entry.
    CHECK_NVML_API(nvmlDeviceGetHandleByIndex(cuda_to_nvml_map(0), &device));

    res = find_own_hostpid(nvml_probe, &device, HOSTPID_DETECT_MAX_ATTEMPTS, &hostpid, &used);
    if (res != NVML_SUCCESS)
        return res;
    LOG_INFO("hostPid=%u", hostpid);
    if (set_host_pid(hostpid) == 0 && used != NVML_VALUE_NOT_AVAILABLE) {
        LOG_INFO("Primary Context Size==%" PRIu64, used);
        context_size = used;
    }
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

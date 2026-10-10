#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/libvgpu.h"
#include "include/libcuda_hook.h"
#include "include/hostpid_fallback_lock.h"
#include "multiprocess/multiprocess_memory_limit.h"

size_t context_size;
fp_dlsym real_dlsym;
static int failures;
static int sizing_locked;
static int saved_pid;
static unsigned int queried_index;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", message);
        failures++;
    }
}

static CUresult device_count(int *count) {
    *count = 1;
    return CUDA_SUCCESS;
}

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END] = {
    [OVERRIDE_cuDeviceGetCount] = {.fn_ptr = device_count},
};

void ensure_initialized(void) {}
int load_env_from_file(char *path) { (void)path; return 0; }
int init_device_info(void) { return 0; }

int set_host_pid(int pid) {
    saved_pid = pid;
    // Production registration calls setspec, which initializes NVML. Keep
    // that side effect so the cold test exercises its real mapping reset.
    check(nvmlInit() == NVML_SUCCESS, "registration initializes fixture NVML");
    return 0;
}

int hostpid_broker_query_trusted(const char *path, pid_t *pid) {
    (void)path;
    *pid = getpid();
    return 0;
}

int hostpid_fallback_lock_deadline_after_ms(struct timespec *deadline,
                                           unsigned int timeout_ms) {
    (void)timeout_ms;
    deadline->tv_sec = 1;
    deadline->tv_nsec = 0;
    return 0;
}

int hostpid_fallback_lock_acquire_shared_until(const struct timespec *deadline) {
    (void)deadline;
    sizing_locked = 1;
    return 0;
}

int hostpid_fallback_lock_release(void) {
    check(sizing_locked, "sizing lock is held until release");
    sizing_locked = 0;
    return 0;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *context, CUdevice device) {
    check(sizing_locked && device == 0, "retain CUDA device zero under lock");
    *context = (CUcontext)1;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice device) {
    check(sizing_locked && device == 0, "release CUDA device zero under lock");
    return CUDA_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *count, nvmlProcessInfo_v1_t *processes) {
    queried_index = (uintptr_t)device - 1;
    // CUDA device 0 is physical device 2. There is no process on physical 0.
    *count = queried_index == 2 ? 1 : 0;
    if (*count != 0) {
        processes[0].pid = getpid();
        processes[0].usedGpuMemory = 1234;
    }
    return NVML_SUCCESS;
}

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "cold") != 0 &&
                      strcmp(argv[1], "warm") != 0)) {
        return 2;
    }
    real_dlsym = dlsym;
    check(setenv("CUDA_VISIBLE_DEVICES", "2", 1) == 0, "set device mapping");
    if (strcmp(argv[1], "warm") == 0) {
        check(nvmlInit() == NVML_SUCCESS, "warm NVML before CUDA initialization");
    }
    // Match postInit's ordering, using the production parser and map storage.
    map_cuda_visible_devices();
    check(cuda_to_nvml_map(0) == 2, "initial map selects physical device two");
    check(set_task_pid_from_broker() == NVML_SUCCESS, "broker lookup succeeds");
    check(saved_pid == getpid(), "broker PID remains registered");
    check(cuda_to_nvml_map(0) == 2 && queried_index == 2,
          "sizing preserves the CUDA to NVML mapping after initialization");
    check(context_size == 1234, "sizing finds the process on its physical GPU");
    check(!sizing_locked, "sizing leaves no node lock held");
    printf("%s NVML: mapping=%u queried=%u context_size=%zu\n",
           argv[1], cuda_to_nvml_map(0), queried_index, context_size);
    if (failures != 0) return 1;
    puts("host PID broker mapping tests passed");
    return 0;
}

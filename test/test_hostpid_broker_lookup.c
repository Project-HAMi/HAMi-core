#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "include/libvgpu.h"
#include "include/libcuda_hook.h"
#include "include/hostpid_broker.h"
#include "include/hostpid_fallback_lock.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define TEST_HOSTPID 424242
#define TEST_MEMORY 123456789ULL
#define HANDLE_BASE 0xC0FFEE00u

static int failures;
static int fake_query_result;
static int fake_query_errno;
static pid_t fake_query_pid;
static nvmlReturn_t fake_nvml_init_result;
static nvmlReturn_t fake_handle_result;
static nvmlReturn_t fake_process_result;
static int fake_not_available;
static int fake_missing_pid;
static int fake_container_pid;
static unsigned int fake_required_count;
static int fake_lock_postinit_count;
static int fake_set_host_pid_count;
static int fake_set_host_pid_value;
static int fake_set_host_pid_result;
static int fake_retain_count;
static int fake_release_count;
static unsigned int fake_handle_index;
static unsigned int fake_process_call_count;
size_t context_size;
int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

static CUresult device_count(int *count) {
    *count = 1;
    return CUDA_SUCCESS;
}

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END] = {
    [OVERRIDE_cuDeviceGetCount] = {.fn_ptr = device_count},
};

static void check(int condition, const char *message) {
    printf("%-4s %s\n", condition ? "OK" : "FAIL", message);
    if (!condition) {
        failures++;
    }
}

int hostpid_broker_query_trusted(const char *socket_path, pid_t *host_pid) {
    check(strcmp(socket_path, HOSTPID_BROKER_SOCKET_PATH) == 0,
          "uses trusted broker socket path");
    if (fake_query_result != 0) {
        *host_pid = 0;
        errno = fake_query_errno;
        return -1;
    }
    *host_pid = fake_query_pid;
    return 0;
}

int set_host_pid(int hostpid) {
    fake_set_host_pid_count++;
    fake_set_host_pid_value = hostpid;
    return fake_set_host_pid_result;
}

static unsigned int fake_mapped_cudadev;

unsigned int cuda_to_nvml_map(unsigned int cudadev) {
    fake_mapped_cudadev = cudadev;
    return 2 + cudadev;
}

unsigned int nvml_to_cuda_map(unsigned int nvmldev) {
    return nvmldev;
}

nvmlReturn_t nvmlInit(void) {
    return fake_nvml_init_result;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index,
                                       nvmlDevice_t *device) {
    fake_handle_index = index;
    *device = (nvmlDevice_t)(uintptr_t)(HANDLE_BASE + index);
    return fake_handle_result;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos) {
    (void)device;
    fake_process_call_count++;
    if (fake_process_result != NVML_SUCCESS) {
        return fake_process_result;
    }
    if (*infoCount < fake_required_count) {
        *infoCount = fake_required_count;
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    for (unsigned int i = 0; i < fake_required_count; i++) {
        infos[i].pid = (unsigned int)(1000U + i);
        infos[i].usedGpuMemory = 10ULL + i;
    }
    infos[fake_required_count - 1U].pid =
        fake_missing_pid ? 999999U
        : fake_container_pid ? (unsigned int)getpid() : TEST_HOSTPID;
    infos[fake_required_count - 1U].usedGpuMemory =
        fake_not_available ? (unsigned long long)NVML_VALUE_NOT_AVAILABLE
                           : TEST_MEMORY;
    *infoCount = fake_required_count;
    return NVML_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    (void)dev;
    fake_retain_count++;
    *pctx = (CUcontext)(uintptr_t)0x1234U;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    (void)dev;
    fake_release_count++;
    return CUDA_SUCCESS;
}

int lock_postinit(void) {
    fake_lock_postinit_count++;
    return 1;
}

static void reset_fakes(void) {
    reset_lazy_context_sizing();
    fake_query_result = 0;
    fake_query_errno = ENOENT;
    fake_query_pid = TEST_HOSTPID;
    fake_nvml_init_result = NVML_SUCCESS;
    fake_handle_result = NVML_SUCCESS;
    fake_process_result = NVML_SUCCESS;
    fake_not_available = 0;
    fake_missing_pid = 0;
    fake_container_pid = 0;
    fake_required_count = 18;
    fake_lock_postinit_count = 0;
    fake_set_host_pid_count = 0;
    fake_set_host_pid_value = 0;
    fake_set_host_pid_result = 0;
    fake_retain_count = 0;
    fake_release_count = 0;
    fake_handle_index = 0;
    fake_mapped_cudadev = 99;
    fake_process_call_count = 0;
    context_size = 0;
}

static void test_success_sets_pid_without_creating_a_context(void) {
    reset_fakes();

    check(set_task_pid_from_broker() == NVML_SUCCESS,
          "broker success returns NVML_SUCCESS");
    check(fake_set_host_pid_count == 1, "broker success sets host PID once");
    check(fake_set_host_pid_value == TEST_HOSTPID,
          "broker success sets queried host PID");
    check(fake_retain_count == 0 && fake_release_count == 0,
          "broker success creates no context during init");
    check(fake_process_call_count == 0, "broker success does not size yet");
    check(context_size == 0, "context_size stays unknown until a retain");
    check(fake_lock_postinit_count == 0,
          "broker lookup does not take the cache postinit lock");
}

static void test_first_retain_sizes_from_own_usage(void) {
    reset_fakes();
    check(set_task_pid_from_broker() == NVML_SUCCESS, "broker lookup succeeds");

    size_primary_context_lazily(1);
    check(context_size == TEST_MEMORY,
          "first retain copies this process's usedGpuMemory to context_size");
    check(fake_mapped_cudadev == 1 && fake_handle_index == 3,
          "sizing reads the retained device through the CUDA-to-NVML map");
    check(fake_process_call_count == 1, "sizing asks NVML once");

    size_primary_context_lazily(0);
    check(fake_process_call_count == 1, "a known size is not measured again");
    check(fake_retain_count == 0, "sizing never creates a context itself");
}

static void test_container_pid_entry_sizes_context(void) {
    reset_fakes();
    fake_container_pid = 1;

    check(set_task_pid_from_broker() == NVML_SUCCESS,
          "container PID entry returns NVML_SUCCESS");
    size_primary_context_lazily(0);
    check(context_size == TEST_MEMORY,
          "driver reporting the container PID still sizes the context");
}

static void test_missing_or_unavailable_usage_keeps_size_unknown(void) {
    for (int scenario = 0; scenario < 2; scenario++) {
        reset_fakes();
        fake_missing_pid = scenario == 0;
        fake_not_available = scenario == 1;
        check(set_task_pid_from_broker() == NVML_SUCCESS,
              "broker lookup succeeds");
        size_primary_context_lazily(0);
        check(context_size == 0, "no usable entry leaves context_size unknown");
        check(fake_set_host_pid_value == TEST_HOSTPID, "the broker PID stays");

        fake_missing_pid = 0;
        fake_not_available = 0;
        size_primary_context_lazily(0);
        check(context_size == TEST_MEMORY, "a later retain can still size");
    }
}

static void test_no_broker_pid_means_no_lazy_sizing(void) {
    reset_fakes();
    fake_query_result = -1;
    check(set_task_pid_from_broker() != NVML_SUCCESS,
          "broker query failure returns an error");
    check(fake_set_host_pid_count == 0, "broker failure does not set host PID");
    size_primary_context_lazily(0);
    check(fake_process_call_count == 0 && context_size == 0,
          "without a broker PID the retain hook does not size");

    reset_fakes();
    fake_set_host_pid_result = -1;
    check(set_task_pid_from_broker() == NVML_ERROR_NOT_FOUND,
          "a PID missing from the shared region is not discovery success");
    size_primary_context_lazily(0);
    check(fake_process_call_count == 0 && context_size == 0,
          "set_host_pid failure disables lazy sizing");

    reset_fakes();
    fake_nvml_init_result = NVML_ERROR_UNINITIALIZED;
    check(set_task_pid_from_broker() == NVML_SUCCESS,
          "NVML init failure keeps the broker PID");
    size_primary_context_lazily(0);
    check(fake_process_call_count == 0 && context_size == 0,
          "NVML init failure disables lazy sizing");
}

static void test_nvml_errors_keep_size_unknown(void) {
    for (int scenario = 0; scenario < 3; scenario++) {
        reset_fakes();
        switch (scenario) {
        case 0:
            fake_handle_result = NVML_ERROR_INVALID_ARGUMENT;
            break;
        case 1:
            fake_process_result = NVML_ERROR_NOT_SUPPORTED;
            break;
        case 2:
            fake_required_count = SHARED_REGION_MAX_PROCESS_NUM + 1U;
            break;
        }
        check(set_task_pid_from_broker() == NVML_SUCCESS,
              "broker lookup succeeds");
        size_primary_context_lazily(0);
        check(context_size == 0, "an NVML error leaves context_size unknown");
        check(fake_process_call_count == (scenario >= 1),
              "sizing stops at the failure and never loops on capacity");
    }
}

static void test_fork_reset_disables_inherited_sizing(void) {
    reset_fakes();
    check(set_task_pid_from_broker() == NVML_SUCCESS, "broker lookup succeeds");
    reset_lazy_context_sizing();
    size_primary_context_lazily(0);
    check(fake_process_call_count == 0 && context_size == 0,
          "a child does not size with its parent's host PID");
}

int main(void) {
    check(setenv("CUDA_VISIBLE_DEVICES", "2", 1) == 0,
          "set a nonidentity device mapping");
    test_success_sets_pid_without_creating_a_context();
    test_first_retain_sizes_from_own_usage();
    test_container_pid_entry_sizes_context();
    test_missing_or_unavailable_usage_keeps_size_unknown();
    test_no_broker_pid_means_no_lazy_sizing();
    test_nvml_errors_keep_size_unknown();
    test_fork_reset_disables_inherited_sizing();

    if (failures != 0) {
        fprintf(stderr, "%d host PID broker lookup checks failed\n", failures);
        return 1;
    }
    puts("host PID broker lookup tests passed");
    return 0;
}

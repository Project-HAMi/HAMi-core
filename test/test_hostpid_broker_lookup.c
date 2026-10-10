#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

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
static CUresult fake_retain_result;
static CUresult fake_release_result;
static int fake_not_available;
static int fake_missing_pid;
static unsigned int fake_required_count;
static int fake_lock_postinit_count;
static int fake_set_host_pid_count;
static int fake_set_host_pid_value;
static int fake_set_host_pid_result;
static int fake_retain_count;
static int fake_release_count;
static unsigned int fake_handle_index;
static unsigned int fake_process_call_count;
static int fake_deadline_result;
static int fake_node_lock_errno;
static int fake_node_unlock_result;
static int fake_node_lock_held;
static int fake_node_unlock_count;
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

unsigned int cuda_to_nvml_map(unsigned int cudadev) {
    check(cudadev == 0, "context lookup asks for CUDA device 0 mapping");
    return 2;
}

unsigned int nvml_to_cuda_map(unsigned int nvmldev) {
    return nvmldev;
}

nvmlReturn_t nvmlInit(void) {
    return fake_nvml_init_result;
}

int hostpid_fallback_lock_deadline_after_ms(struct timespec *deadline,
                                           unsigned int timeout_ms) {
    check(timeout_ms == HOSTPID_FALLBACK_LOCK_TIMEOUT_MS,
          "sizing uses the configured lock timeout");
    deadline->tv_sec = 123;
    deadline->tv_nsec = 456;
    errno = EIO;
    return fake_deadline_result;
}

int hostpid_fallback_lock_acquire_shared_until(const struct timespec *deadline) {
    check(deadline->tv_sec == 123 && deadline->tv_nsec == 456,
          "sizing lock uses the computed deadline");
    if (fake_node_lock_errno != 0) {
        errno = fake_node_lock_errno;
        return -1;
    }
    fake_node_lock_held = 1;
    return 0;
}

int hostpid_fallback_lock_release(void) {
    check(fake_node_lock_held, "only an acquired sizing lock is released");
    fake_node_lock_held = 0;
    fake_node_unlock_count++;
    errno = EIO;
    return fake_node_unlock_result;
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
    check(fake_node_lock_held, "NVML sizing holds the shared node lock");
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
        fake_missing_pid ? 999999U : TEST_HOSTPID;
    infos[fake_required_count - 1U].usedGpuMemory =
        fake_not_available ? (unsigned long long)NVML_VALUE_NOT_AVAILABLE
                           : TEST_MEMORY;
    *infoCount = fake_required_count;
    return NVML_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    check(fake_node_lock_held, "context retain holds the shared node lock");
    check(dev == 0, "retains CUDA device 0 primary context");
    fake_retain_count++;
    *pctx = (CUcontext)(uintptr_t)0x1234U;
    return fake_retain_result;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    check(fake_node_lock_held, "context release holds the shared node lock");
    check(dev == 0, "releases CUDA device 0 primary context");
    fake_release_count++;
    return fake_release_result;
}

int lock_postinit(void) {
    fake_lock_postinit_count++;
    return 1;
}

static void reset_fakes(void) {
    check(!fake_node_lock_held, "previous lookup left no sizing lock held");
    fake_deadline_result = 0;
    fake_node_lock_errno = 0;
    fake_node_unlock_result = 0;
    fake_node_unlock_count = 0;
    fake_query_result = 0;
    fake_query_errno = ENOENT;
    fake_query_pid = TEST_HOSTPID;
    fake_nvml_init_result = NVML_SUCCESS;
    fake_handle_result = NVML_SUCCESS;
    fake_process_result = NVML_SUCCESS;
    fake_retain_result = CUDA_SUCCESS;
    fake_release_result = CUDA_SUCCESS;
    fake_not_available = 0;
    fake_missing_pid = 0;
    fake_required_count = 18;
    fake_lock_postinit_count = 0;
    fake_set_host_pid_count = 0;
    fake_set_host_pid_value = 0;
    fake_set_host_pid_result = 0;
    fake_retain_count = 0;
    fake_release_count = 0;
    fake_handle_index = 0;
    fake_process_call_count = 0;
    context_size = 0;
}

static void test_success_sets_pid_context_size_and_balances_context(void) {
    reset_fakes();

    check(set_task_pid_from_broker() == NVML_SUCCESS,
          "broker success returns NVML_SUCCESS");
    check(fake_set_host_pid_count == 1, "broker success sets host PID once");
    check(fake_set_host_pid_value == TEST_HOSTPID,
          "broker success sets queried host PID");
    check(context_size == TEST_MEMORY,
          "broker success copies matching usedGpuMemory to context_size");
    check(fake_retain_count == 1, "broker success retains once");
    check(fake_release_count == 1, "broker success releases once");
    check(fake_handle_index == 2, "context lookup uses CUDA-to-NVML map");
    check(fake_process_call_count == 1, "context lookup asks NVML once");
    check(fake_node_unlock_count == 1, "success releases the sizing lock");
    check(fake_lock_postinit_count == 0,
          "broker lookup does not take the cache postinit lock");
}

static void test_missing_entry_keeps_pid_and_context_size(void) {
    reset_fakes();
    fake_missing_pid = 1;
    context_size = 999;

    check(set_task_pid_from_broker() == NVML_SUCCESS,
          "missing entry still returns NVML_SUCCESS after broker success");
    check(fake_set_host_pid_count == 1, "missing entry keeps broker PID");
    check(context_size == 999, "missing entry leaves context_size unchanged");
    check(fake_retain_count == 1, "missing entry retains once");
    check(fake_release_count == 1, "missing entry releases once");
}

static void test_not_available_keeps_pid_and_context_size(void) {
    reset_fakes();
    fake_not_available = 1;
    context_size = 999;

    check(set_task_pid_from_broker() == NVML_SUCCESS,
          "NOT_AVAILABLE entry still returns NVML_SUCCESS after broker success");
    check(fake_set_host_pid_count == 1, "NOT_AVAILABLE keeps broker PID");
    check(context_size == 999, "NOT_AVAILABLE leaves context_size unchanged");
    check(fake_retain_count == 1, "NOT_AVAILABLE retains once");
    check(fake_release_count == 1, "NOT_AVAILABLE releases once");
}

static void test_broker_failure_returns_error_and_makes_no_retain(void) {
    reset_fakes();
    fake_query_result = -1;

    check(set_task_pid_from_broker() != NVML_SUCCESS,
          "broker query failure returns an error");
    check(fake_set_host_pid_count == 0, "broker failure does not set host PID");
    check(fake_retain_count == 0, "broker failure does not retain context");
    check(fake_release_count == 0, "broker failure does not release context");
    check(context_size == 0, "broker failure leaves context_size zero");
}

static void test_set_host_pid_failure_skips_context_lookup(void) {
    reset_fakes();
    fake_set_host_pid_result = -1;

    check(set_task_pid_from_broker() == NVML_ERROR_NOT_FOUND,
          "a PID missing from the shared region is not discovery success");
    check(fake_retain_count == 0, "set_host_pid failure does not retain");
    check(context_size == 0, "set_host_pid failure leaves context_size zero");
}

static void test_sizing_errors_keep_pid_and_release_retained_context(void) {
    for (int scenario = 0; scenario < 5; scenario++) {
        reset_fakes();
        context_size = 999;
        switch (scenario) {
        case 0:
            fake_nvml_init_result = NVML_ERROR_UNINITIALIZED;
            break;
        case 1:
            fake_retain_result = CUDA_ERROR_OUT_OF_MEMORY;
            break;
        case 2:
            fake_handle_result = NVML_ERROR_INVALID_ARGUMENT;
            break;
        case 3:
            fake_process_result = NVML_ERROR_NOT_SUPPORTED;
            break;
        case 4:
            fake_required_count = SHARED_REGION_MAX_PROCESS_NUM + 1U;
            break;
        }
        check(set_task_pid_from_broker() == NVML_SUCCESS,
              "sizing failure preserves broker discovery success");
        check(fake_set_host_pid_value == TEST_HOSTPID,
              "sizing failure keeps the broker PID");
        check(context_size == 999, "sizing failure preserves context_size");
        check(fake_retain_count == (scenario != 0),
              "NVML init failure skips the retain");
        check(fake_release_count == (scenario >= 2),
              "sizing failure releases only a successfully retained context");
        check(fake_process_call_count == (scenario >= 3),
              "sizing stops at failure and never loops on capacity");
        check(fake_node_unlock_count == (scenario != 0),
              "sizing failures release an acquired node lock");
    }
}

static void test_lock_errors_keep_pid_without_creating_context(void) {
    static const int errors[] = {EACCES, ENOENT, ETIMEDOUT};
    for (size_t i = 0; i <= sizeof(errors) / sizeof(errors[0]); i++) {
        reset_fakes();
        context_size = 999;
        if (i == 0) {
            fake_deadline_result = -1;
        } else {
            fake_node_lock_errno = errors[i - 1];
        }
        check(set_task_pid_from_broker() == NVML_SUCCESS,
              "lock failure keeps the known broker PID");
        check(fake_set_host_pid_value == TEST_HOSTPID,
              "lock failure preserves registration");
        check(fake_retain_count == 0 && fake_process_call_count == 0,
              "lock failure never creates a context or sizes it");
        check(context_size == 999 && fake_node_unlock_count == 0,
              "lock failure keeps size and does not unlock");
    }
}

static void test_release_failure_preserves_pid_and_size(void) {
    reset_fakes();
    fake_release_result = CUDA_ERROR_INVALID_CONTEXT;
    check(set_task_pid_from_broker() == NVML_SUCCESS,
          "release failure does not discard the broker PID");
    check(fake_set_host_pid_value == TEST_HOSTPID, "release failure keeps PID");
    check(context_size == TEST_MEMORY, "release failure keeps measured size");
    check(fake_release_count == 1, "failed release is attempted once");
    check(fake_node_unlock_count == 1, "CUDA release failure unlocks node");

    reset_fakes();
    fake_node_unlock_result = -1;
    check(set_task_pid_from_broker() == NVML_SUCCESS,
          "node unlock failure does not discard the broker PID");
    check(context_size == TEST_MEMORY && fake_node_unlock_count == 1,
          "node unlock failure preserves size and is attempted once");
}

int main(void) {
    check(setenv("CUDA_VISIBLE_DEVICES", "2", 1) == 0,
          "set a nonidentity device mapping");
    test_success_sets_pid_context_size_and_balances_context();
    test_missing_entry_keeps_pid_and_context_size();
    test_not_available_keeps_pid_and_context_size();
    test_broker_failure_returns_error_and_makes_no_retain();
    test_set_host_pid_failure_skips_context_lookup();
    test_sizing_errors_keep_pid_and_release_retained_context();
    test_lock_errors_keep_pid_without_creating_context();
    test_release_failure_preserves_pid_and_size();

    if (failures != 0) {
        fprintf(stderr, "%d host PID broker lookup checks failed\n", failures);
        return 1;
    }
    puts("host PID broker lookup tests passed");
    return 0;
}

/*
 * GPU-free regression test for get_used_gpu_utilization() (issue #318).
 * The watcher must ask NVML for the NVML index it is walking, not the CUDA
 * index it maps to; the two only coincide when CUDA_VISIBLE_DEVICES is the
 * identity map.  NVML and the shared region are stubbed below.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/nvml_prefix.h"
#include <nvml.h>  // NOLINT(build/include_order)

#include "multiprocess/multiprocess_memory_limit.h"
#include "include/log_utils.h"
#include "include/nvml_processes_utilization_subset.h"

int get_used_gpu_utilization(int *userutil, int *sysprocnum);
extern int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

typedef unsigned long long nvml_ull;  // NOLINT(runtime/int)

#define MAX_FAKE_DEVICES 4
#define HANDLE_BASE 0xD1CE0000u
#define TEST_HOSTPID 4242
#define MIB (1024ULL * 1024ULL)

static unsigned int fake_device_count;
static unsigned int fake_sm_util[MAX_FAKE_DEVICES];
static nvml_ull fake_used_mem[MAX_FAKE_DEVICES];
static unsigned int handle_requests[MAX_FAKE_DEVICES + 1];
static unsigned int handle_request_count;
static shrreg_proc_slot_t test_slot;

static unsigned int index_of(nvmlDevice_t device) {
    return (unsigned int)(uintptr_t)device - HANDLE_BASE;
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    *deviceCount = fake_device_count;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
    *deviceCount = fake_device_count;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index,
                                        nvmlDevice_t *device) {
    handle_request_count++;
    if (index >= fake_device_count) {
        handle_requests[MAX_FAKE_DEVICES]++;
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    handle_requests[index]++;
    *device = (nvmlDevice_t)(uintptr_t)(HANDLE_BASE + index);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device,
                                                  unsigned int *infoCount,
                                                  nvmlProcessInfo_v1_t *infos) {
    unsigned int idx = index_of(device);
    if (*infoCount < 1) {
        *infoCount = 1;
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    infos[0].pid = TEST_HOSTPID;
    infos[0].usedGpuMemory = fake_used_mem[idx];
    *infoCount = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetProcessUtilization(
    nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization,
    unsigned int *processSamplesCount, nvml_ull lastSeenTimeStamp) {
    unsigned int idx = index_of(device);
    (void)lastSeenTimeStamp;
    if (*processSamplesCount < 1) {
        *processSamplesCount = 1;
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    memset(&utilization[0], 0, sizeof(utilization[0]));
    utilization[0].pid = TEST_HOSTPID;
    utilization[0].smUtil = fake_sm_util[idx];
    *processSamplesCount = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetProcessesUtilizationInfo(
    nvmlDevice_t device, nvmlProcessesUtilizationInfo_t *procesesUtilInfo) {
    (void)device;
    (void)procesesUtilInfo;
    return NVML_ERROR_NOT_SUPPORTED;
}

void lock_shrreg(void) {}
void unlock_shrreg(void) {}

shrreg_proc_slot_t *find_proc_by_hostpid(int hostpid) {
    return hostpid == TEST_HOSTPID ? &test_slot : NULL;
}

static int failures;

static void check(int ok, const char *what) {
    printf("%-4s %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

static void set_visible_devices(const unsigned int *visible, unsigned int n) {
    for (int i = 0; i < CUDA_DEVICE_MAX_COUNT; i++) {
        cuda_to_nvml_map_array[i] = i;
    }
    for (unsigned int i = 0; i < n; i++) {
        cuda_to_nvml_map_array[i] = visible[i];
    }
}

/* Mirrors nvml_to_cuda_map(): an NVML device is sampled when it appears in the
 * first device_count map entries.  Entries beyond the CUDA_VISIBLE_DEVICES list
 * keep the identity defaults set by parse_cuda_visible_env(), so with
 * CUDA_VISIBLE_DEVICES=1 on four GPUs, NVML 2 and 3 are still sampled.  That is
 * pre-existing behaviour, not changed by the #318 fix; the test records it. */
static int is_mapped(unsigned int nvml, unsigned int device_count) {
    for (unsigned int cuda = 0; cuda < device_count; cuda++) {
        unsigned int entry = cuda_to_nvml_map_array[cuda];
        if (entry == nvml) {
            return 1;
        }
    }
    return 0;
}

static void reset_fakes(unsigned int device_count) {
    fake_device_count = device_count;
    for (unsigned int i = 0; i < MAX_FAKE_DEVICES; i++) {
        fake_sm_util[i] = 10u * (i + 1);
        fake_used_mem[i] = (i + 1) * 100ULL * MIB;
    }
    memset(handle_requests, 0, sizeof(handle_requests));
    handle_request_count = 0;
    memset(&test_slot, 0, sizeof(test_slot));
    test_slot.hostpid = TEST_HOSTPID;
}

static void run_case(const char *name, unsigned int device_count,
                     const unsigned int *visible, unsigned int n) {
    int userutil[CUDA_DEVICE_MAX_COUNT];
    int sysprocnum = 0;
    char what[160];

    printf("--- %s ---\n", name);
    reset_fakes(device_count);
    set_visible_devices(visible, n);
    memset(userutil, 0, sizeof(userutil));

    int rc = get_used_gpu_utilization(userutil, &sysprocnum);
    check(rc == 0, "get_used_gpu_utilization returns 0");

    unsigned int expected_requests = 0;
    for (unsigned int nvml = 0; nvml < device_count; nvml++) {
        expected_requests += is_mapped(nvml, device_count) ? 1u : 0u;
    }
    snprintf(what, sizeof(what), "asked NVML for %u handles, one per mapped device", expected_requests);
    check(handle_request_count == expected_requests, what);

    for (unsigned int nvml = 0; nvml < device_count; nvml++) {
        int is_visible = 0;
        for (unsigned int cuda = 0; cuda < n; cuda++) {
            if (visible[cuda] == nvml) {
                is_visible = 1;
            }
        }
        int mapped = is_mapped(nvml, device_count);
        printf("  NVML %u: %s, handle requested %u time(s)\n", nvml,
               is_visible ? "visible"
               : mapped ? "leftover identity entry, sampled (pre-existing behaviour)"
               : "not mapped",
               handle_requests[nvml]);
        snprintf(what, sizeof(what), mapped
                 ? "NVML %u: handle requested exactly once"
                 : "NVML %u: handle never requested", nvml);
        check(handle_requests[nvml] == (mapped ? 1u : 0u), what);
    }

    for (unsigned int cuda = 0; cuda < n; cuda++) {
        unsigned int nvml = visible[cuda];
        int want_util = fake_sm_util[nvml];

        printf("  CUDA %u -> NVML %u: userutil=%d (want %d)\n",
               cuda, nvml, userutil[cuda], want_util);

        snprintf(what, sizeof(what), "CUDA %u: userutil comes from NVML device %u", cuda, nvml);
        check(userutil[cuda] == want_util, what);

        snprintf(what, sizeof(what), "CUDA %u: per-process sm_util comes from NVML device %u", cuda, nvml);
        check(test_slot.device_util[cuda].sm_util == fake_sm_util[nvml], what);

        snprintf(what, sizeof(what), "CUDA %u: monitorused comes from NVML device %u", cuda, nvml);
        check(test_slot.monitorused[cuda] == fake_used_mem[nvml], what);
    }
}

int main(void) {
    log_utils_init();

    static const unsigned int subset[] = {2, 3};
    run_case("CUDA_VISIBLE_DEVICES=2,3 on a 4-GPU node", 4, subset, 2);

    static const unsigned int single[] = {1};
    run_case("CUDA_VISIBLE_DEVICES=1 on a 4-GPU node", 4, single, 1);

    static const unsigned int reversed[] = {1, 0};
    run_case("CUDA_VISIBLE_DEVICES=1,0 on a 2-GPU node", 2, reversed, 2);

    static const unsigned int identity[] = {0, 1, 2};
    run_case("CUDA_VISIBLE_DEVICES=0,1,2 on a 3-GPU node", 3, identity, 3);

    printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

// Copyright 2026 The HAMi Authors.
/*
 * GPU-free unit test for CUDA module VRAM accounting (#400):
 * - cuModuleLoad / cuModuleLoadData / cuModuleLoadDataEx / cuModuleLoadFatBinary
 * - Concurrent data allocation cancellation during NVML delta sampling
 * - Quota admission failure (OOM) rollback via native cuModuleUnload
 * - cuModuleUnload charge release
 * - Context teardown cleanup (cuCtxDestroy_v2, cuDevicePrimaryCtxRelease_v2,
 *   cuDevicePrimaryCtxReset_v2)
 * - Fork child reset without perturbing parent shared-region accounting
 */
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "allocator/allocator.h"
#include "include/hostpid_broker.h"
#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "include/utils.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define MIB (UINT64_C(1) << 20)
#define TEST_DEV 0

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];

void ensure_hostpid(void) {
    shrreg_proc_slot_t *slot = find_proc_by_hostpid(getpid());
    if (slot != NULL) {
        atomic_store_explicit(&slot->hostpid, getpid(), memory_order_release);
    }
}

void size_primary_context_lazily(int cudadev) {
    (void)cudadev;
}

int driver_admission_enabled(void) {
    return 0;
}

int driver_admission_acquire(hostpid_driver_admission_operation_t operation) {
    (void)operation;
    return -1;
}

void driver_admission_release(int *lease_fd) {
    *lease_fd = -1;
}

void driver_admission_cleanup(void *lease_fd) {
    driver_admission_release(lease_fd);
}

unsigned int cuda_to_nvml_map(unsigned int cudadev) {
    return cudadev;
}

typedef struct {
    uintptr_t handle;
    size_t size;
    int live;
} fake_module_t;

static fake_module_t fake_modules[64];
static uintptr_t next_mod_handle = 0x1000;
static _Atomic uint64_t fake_nvml_used;
static size_t next_module_size;
static size_t concurrent_data_alloc_during_load;
static size_t spawn_inflight_during_load;
static pthread_t midload_worker;
static int midload_worker_active;
static pthread_mutex_t inflight_gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t inflight_cond = PTHREAD_COND_INITIALIZER;
static int inflight_reserved;
static int unload_calls;

static void *inflight_reservation_worker(void *arg) {
    size_t alloc_size = *(const size_t *)arg;
    memory_op_enter();
    if (reserve_device_memory(TEST_DEV, alloc_size) != 0) {
        memory_op_leave();
        return (void *)(uintptr_t)1;
    }
    pthread_mutex_lock(&inflight_gate);
    inflight_reserved = 1;
    pthread_cond_broadcast(&inflight_cond);
    pthread_mutex_unlock(&inflight_gate);

    usleep(20000);
    fake_nvml_used += alloc_size;
    memory_op_leave();
    return NULL;
}

static CUcontext current_ctx;
static CUcontext primary_ctx = (CUcontext)(uintptr_t)0xabc0;
static cuuint64_t primary_ctx_gen = 10;
static int primary_refcount;

int get_process_gpu_memory(pid_t hostpid, int cudadev, uint64_t *used) {
    if (hostpid != getpid() || cudadev != TEST_DEV || used == NULL) {
        return -1;
    }
    *used = fake_nvml_used;
    return 0;
}

static CUresult stub_ctx_get_current(CUcontext *pctx) {
    *pctx = current_ctx;
    return CUDA_SUCCESS;
}

static CUresult stub_ctx_get_device(CUdevice *dev) {
    if (current_ctx == NULL) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *dev = TEST_DEV;
    return CUDA_SUCCESS;
}

static CUresult stub_ctx_get_id(CUcontext ctx, cuuint64_t *ctx_id) {
    if (ctx == NULL) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (ctx == primary_ctx) {
        *ctx_id = primary_ctx_gen;
    } else {
        *ctx_id = (cuuint64_t)(uintptr_t)ctx;
    }
    return CUDA_SUCCESS;
}

static CUresult alloc_fake_module(CUmodule *module) {
    if (concurrent_data_alloc_during_load > 0) {
        add_gpu_device_memory_usage(getpid(), TEST_DEV,
                                    concurrent_data_alloc_during_load, 2);
        fake_nvml_used += concurrent_data_alloc_during_load;
    }
    if (spawn_inflight_during_load > 0) {
        pthread_mutex_lock(&inflight_gate);
        inflight_reserved = 0;
        pthread_mutex_unlock(&inflight_gate);
        if (pthread_create(&midload_worker, NULL, inflight_reservation_worker,
                           &spawn_inflight_during_load) != 0) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        midload_worker_active = 1;
        pthread_mutex_lock(&inflight_gate);
        while (!inflight_reserved) {
            pthread_cond_wait(&inflight_cond, &inflight_gate);
        }
        pthread_mutex_unlock(&inflight_gate);
    }
    for (size_t i = 0; i < sizeof(fake_modules) / sizeof(fake_modules[0]); i++) {
        if (!fake_modules[i].live) {
            uintptr_t h = next_mod_handle++;
            fake_modules[i].handle = h;
            fake_modules[i].size = next_module_size;
            fake_modules[i].live = 1;
            fake_nvml_used += next_module_size;
            *module = (CUmodule)h;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}

static CUresult stub_module_load(CUmodule *module, const char *fname) {
    (void)fname;
    return alloc_fake_module(module);
}

static CUresult stub_module_load_data(CUmodule *module, const void *image) {
    (void)image;
    return alloc_fake_module(module);
}

static CUresult stub_module_load_data_ex(CUmodule *module, const void *image,
                                         unsigned int numOptions,
                                         CUjit_option *options,
                                         void **optionValues) {
    (void)image;
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return alloc_fake_module(module);
}

static CUresult stub_module_load_fat_binary(CUmodule *module,
                                            const void *fatCubin) {
    (void)fatCubin;
    return alloc_fake_module(module);
}

static CUresult stub_module_unload(CUmodule hmod) {
    unload_calls++;
    uintptr_t h = (uintptr_t)hmod;
    for (size_t i = 0; i < sizeof(fake_modules) / sizeof(fake_modules[0]); i++) {
        if (fake_modules[i].live && fake_modules[i].handle == h) {
            fake_nvml_used -= fake_modules[i].size;
            fake_modules[i].live = 0;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

static CUresult stub_ctx_destroy(CUcontext ctx) {
    if (ctx == NULL) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (current_ctx == ctx) {
        current_ctx = NULL;
    }
    return CUDA_SUCCESS;
}

static CUresult stub_primary_ctx_retain(CUcontext *pctx, CUdevice dev) {
    if (dev != TEST_DEV) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    primary_refcount++;
    *pctx = primary_ctx;
    current_ctx = primary_ctx;
    return CUDA_SUCCESS;
}

static CUresult stub_primary_ctx_release(CUdevice dev) {
    if (dev != TEST_DEV || primary_refcount <= 0) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    primary_refcount--;
    if (primary_refcount == 0) {
        primary_ctx_gen++;
    }
    return CUDA_SUCCESS;
}

static CUresult stub_primary_ctx_get_state(CUdevice dev, unsigned int *flags,
                                           int *active) {
    if (dev != TEST_DEV) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *flags = 0;
    *active = (primary_refcount > 0) ? 1 : 0;
    return CUDA_SUCCESS;
}

static CUresult stub_primary_ctx_reset(CUdevice dev) {
    if (dev != TEST_DEV) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    primary_ctx_gen++;
    return CUDA_SUCCESS;
}

static uint64_t current_module_size(void) {
    shrreg_proc_slot_t *slot = find_proc_by_hostpid(getpid());
    if (slot == NULL) {
        return UINT64_MAX;
    }
    return atomic_load_explicit(&slot->used[TEST_DEV].module_size,
                                memory_order_acquire);
}

static void init_stub_table(void) {
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuCtxGetCurrent)].fn_ptr =
        (void *)stub_ctx_get_current;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuCtxGetDevice)].fn_ptr =
        (void *)stub_ctx_get_device;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuCtxGetId)].fn_ptr =
        (void *)stub_ctx_get_id;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuModuleLoad)].fn_ptr =
        (void *)stub_module_load;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuModuleLoadData)].fn_ptr =
        (void *)stub_module_load_data;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuModuleLoadDataEx)].fn_ptr =
        (void *)stub_module_load_data_ex;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuModuleLoadFatBinary)].fn_ptr =
        (void *)stub_module_load_fat_binary;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuModuleUnload)].fn_ptr =
        (void *)stub_module_unload;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuCtxDestroy_v2)].fn_ptr =
        (void *)stub_ctx_destroy;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRetain)].fn_ptr =
        (void *)stub_primary_ctx_retain;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRelease_v2)].fn_ptr =
        (void *)stub_primary_ctx_release;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxGetState)].fn_ptr =
        (void *)stub_primary_ctx_get_state;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxReset_v2)].fn_ptr =
        (void *)stub_primary_ctx_reset;
}

static int run_tests(void) {
    CUcontext ctx_a = (CUcontext)(uintptr_t)0x100;
    CUcontext ctx_b = (CUcontext)(uintptr_t)0x200;
    CUmodule m1 = NULL, m2 = NULL, m3 = NULL, m4 = NULL;

    REQUIRE(set_current_device_memory_limit(TEST_DEV, 512 * MIB) == 0);
    current_ctx = ctx_a;

    /* 1. All four cuModuleLoad* entry points charge module_size and unload releases. */
    next_module_size = 64 * MIB;
    REQUIRE(cuModuleLoad(&m1, "a.cubin") == CUDA_SUCCESS && m1 != NULL);
    next_module_size = 96 * MIB;
    REQUIRE(cuModuleLoadData(&m2, "ptx") == CUDA_SUCCESS && m2 != NULL);
    next_module_size = 32 * MIB;
    REQUIRE(cuModuleLoadDataEx(&m3, "ptx", 0, NULL, NULL) == CUDA_SUCCESS && m3 != NULL);
    next_module_size = 48 * MIB;
    REQUIRE(cuModuleLoadFatBinary(&m4, "fatbin") == CUDA_SUCCESS && m4 != NULL);
    REQUIRE(current_module_size() == 240 * MIB);
    REQUIRE(get_gpu_memory_usage(TEST_DEV) == 240 * MIB);

    REQUIRE(cuModuleUnload(m1) == CUDA_SUCCESS);
    REQUIRE(cuModuleUnload(m2) == CUDA_SUCCESS);
    REQUIRE(cuModuleUnload(m3) == CUDA_SUCCESS);
    REQUIRE(cuModuleUnload(m4) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);
    REQUIRE(get_gpu_memory_usage(TEST_DEV) == 0);

    /* 2. Concurrent tracked data allocation during load cancels out cleanly. */
    next_module_size = 128 * MIB;
    concurrent_data_alloc_during_load = 200 * MIB;
    REQUIRE(cuModuleLoadData(&m1, "ptx") == CUDA_SUCCESS);
    concurrent_data_alloc_during_load = 0;
    REQUIRE(current_module_size() == 128 * MIB);
    REQUIRE(get_gpu_memory_usage(TEST_DEV) == 328 * MIB);
    rm_gpu_device_memory_usage(getpid(), TEST_DEV, 200 * MIB, 2);
    fake_nvml_used -= 200 * MIB;
    REQUIRE(cuModuleUnload(m1) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);
    REQUIRE(get_gpu_memory_usage(TEST_DEV) == 0);

    /* 2b. Reservation still in flight when module load starts is drained first. */
    size_t inflight_bytes = 200 * MIB;
    pthread_t worker;
    void *worker_ret = NULL;
    pthread_mutex_lock(&inflight_gate);
    inflight_reserved = 0;
    pthread_mutex_unlock(&inflight_gate);
    REQUIRE(pthread_create(&worker, NULL, inflight_reservation_worker,
                           &inflight_bytes) == 0);
    pthread_mutex_lock(&inflight_gate);
    while (!inflight_reserved) {
        pthread_cond_wait(&inflight_cond, &inflight_gate);
    }
    pthread_mutex_unlock(&inflight_gate);

    next_module_size = 128 * MIB;
    REQUIRE(cuModuleLoadData(&m1, "ptx_inflight_pre") == CUDA_SUCCESS);
    REQUIRE(pthread_join(worker, &worker_ret) == 0 && worker_ret == NULL);
    REQUIRE(current_module_size() == 128 * MIB);
    REQUIRE(get_gpu_memory_usage(TEST_DEV) == 328 * MIB);
    release_device_memory(TEST_DEV, inflight_bytes);
    fake_nvml_used -= inflight_bytes;
    REQUIRE(cuModuleUnload(m1) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);
    REQUIRE(get_gpu_memory_usage(TEST_DEV) == 0);

    /* 2c. Reservation still in flight before post-load sample is drained first. */
    next_module_size = 128 * MIB;
    spawn_inflight_during_load = 160 * MIB;
    midload_worker_active = 0;
    REQUIRE(cuModuleLoadData(&m1, "ptx_inflight_post") == CUDA_SUCCESS);
    REQUIRE(midload_worker_active == 1);
    REQUIRE(pthread_join(midload_worker, &worker_ret) == 0 && worker_ret == NULL);
    midload_worker_active = 0;
    REQUIRE(current_module_size() == 128 * MIB);
    REQUIRE(get_gpu_memory_usage(TEST_DEV) == 288 * MIB);
    release_device_memory(TEST_DEV, spawn_inflight_during_load);
    fake_nvml_used -= spawn_inflight_during_load;
    spawn_inflight_during_load = 0;
    REQUIRE(cuModuleUnload(m1) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);
    REQUIRE(get_gpu_memory_usage(TEST_DEV) == 0);

    /* 3. Quota exceeded rolls back via native cuModuleUnload and returns OOM. */
    next_module_size = 300 * MIB;
    REQUIRE(cuModuleLoadData(&m1, "fits") == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 300 * MIB);

    int prev_unloads = unload_calls;
    m2 = (CUmodule)(uintptr_t)0xdead;
    REQUIRE(cuModuleLoadData(&m2, "overflows") == CUDA_ERROR_OUT_OF_MEMORY);
    REQUIRE(m2 == NULL);
    REQUIRE(unload_calls == prev_unloads + 1);
    REQUIRE(fake_nvml_used == 300 * MIB);
    REQUIRE(current_module_size() == 300 * MIB);
    REQUIRE(cuModuleUnload(m1) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);

    /* 4. cuCtxDestroy_v2 releases modules in that context only. */
    current_ctx = ctx_a;
    next_module_size = 120 * MIB;
    REQUIRE(cuModuleLoadData(&m1, "ctx_a_mod") == CUDA_SUCCESS);
    current_ctx = ctx_b;
    next_module_size = 80 * MIB;
    REQUIRE(cuModuleLoadData(&m2, "ctx_b_mod") == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 200 * MIB);

    fake_nvml_used -= 120 * MIB;
    REQUIRE(cuCtxDestroy_v2(ctx_a) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 80 * MIB);
    fake_nvml_used -= 80 * MIB;
    REQUIRE(cuCtxDestroy_v2(ctx_b) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);

    /* 5. Primary context release (on last ref) and reset clean up modules. */
    CUcontext retained = NULL;
    REQUIRE(cuDevicePrimaryCtxRetain(&retained, TEST_DEV) == CUDA_SUCCESS);
    REQUIRE(cuDevicePrimaryCtxRetain(&retained, TEST_DEV) == CUDA_SUCCESS);
    next_module_size = 150 * MIB;
    REQUIRE(cuModuleLoadData(&m1, "primary_mod") == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 150 * MIB);

    REQUIRE(cuDevicePrimaryCtxRelease_v2(TEST_DEV) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 150 * MIB);
    fake_nvml_used -= 150 * MIB;
    REQUIRE(cuDevicePrimaryCtxRelease_v2(TEST_DEV) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);

    REQUIRE(cuDevicePrimaryCtxRetain(&retained, TEST_DEV) == CUDA_SUCCESS);
    next_module_size = 110 * MIB;
    REQUIRE(cuModuleLoadData(&m1, "reset_mod") == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 110 * MIB);
    fake_nvml_used -= 110 * MIB;
    REQUIRE(cuDevicePrimaryCtxReset_v2(TEST_DEV) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);

    next_module_size = 90 * MIB;
    REQUIRE(cuModuleLoadData(&m1, "post_reset_mod") == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 90 * MIB);
    fake_nvml_used -= 90 * MIB;
    REQUIRE(cuDevicePrimaryCtxRelease_v2(TEST_DEV) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);

    /* 6. Forked child drops inherited entries automatically via pthread_atfork. */
    current_ctx = ctx_a;
    next_module_size = 140 * MIB;
    REQUIRE(cuModuleLoadData(&m1, "parent_mod") == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 140 * MIB);

    pid_t child = fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        module_release_unloaded(m1);
        _exit(0);
    }
    int status = 0;
    REQUIRE(waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    REQUIRE(current_module_size() == 140 * MIB);
    REQUIRE(cuModuleUnload(m1) == CUDA_SUCCESS);
    REQUIRE(current_module_size() == 0);

    return 0;
}

int main(void) {
    char cache_path[128];
    snprintf(cache_path, sizeof(cache_path),
             "/tmp/test_module_accounting_%d.cache", (int)getpid());
    setenv("CUDA_DEVICE_MEMORY_SHARED_CACHE", cache_path, 1);

    init_stub_table();
    ensure_initialized();
    ensure_hostpid();

    int rc = run_tests();
    unlink(cache_path);
    if (rc == 0) {
        printf("PASS\n");
    }
    return rc;
}

// Copyright 2026 The HAMi Authors.
/*
 * GPU-free concurrency regression test for device_overallocated locking
 * across add_chunk_only, remove_chunk, remove_chunk_only (via VMM release),
 * and check_memory_type (issue #445).
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"

#define ITERATIONS 2000
#define CHUNK_SIZE 4096

void allocator_init(void);

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];

static atomic_size_t removed_bytes;
static atomic_int stop_reader;
static int worker_error;
static int failures;

#define EXPECT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); \
        failures++; \
    } \
} while (0)

int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t usage, int type) {
    (void)pid;
    (void)dev;
    (void)type;
    atomic_fetch_add_explicit(&removed_bytes, usage, memory_order_relaxed);
    return 0;
}

int add_gpu_device_memory_usage(int32_t pid, int dev, size_t usage, int type) {
    (void)pid;
    (void)dev;
    (void)usage;
    (void)type;
    return 0;
}

CUresult cuCtxGetDevice(CUdevice *dev) {
    *dev = 0;
    return CUDA_SUCCESS;
}

uint64_t get_current_device_memory_limit(const int dev) {
    (void)dev;
    return 0;
}

size_t get_gpu_memory_usage(const int dev) {
    (void)dev;
    return 0;
}

unsigned int cuda_to_nvml_map(unsigned int cudadev) { return cudadev; }
void lock_shrreg(void) {}
void unlock_shrreg(void) {}
int clear_proc_slot_nolock(int do_clear) {
    (void)do_clear;
    return 0;
}

CUresult cuMemoryAllocate(CUdeviceptr *dptr, size_t bytesize, void *data) {
    (void)bytesize;
    (void)data;
    *dptr = 0x1000;
    return CUDA_SUCCESS;
}

CUresult cuMemoryFree(CUdeviceptr dptr) {
    (void)dptr;
    return CUDA_SUCCESS;
}

static void *worker_alloc_free(void *arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        CUdeviceptr ptr = (CUdeviceptr)(0x100000ULL + (uint64_t)i * CHUNK_SIZE);
        if (add_chunk_only(ptr, CHUNK_SIZE, 0) != 0) {
            return &worker_error;
        }
        if (free_raw(ptr) != 0) {
            return &worker_error;
        }
    }
    return NULL;
}

static void *worker_vmm_lifecycle(void *arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        CUmemGenericAllocationHandle h =
            (CUmemGenericAllocationHandle)(0x80000000ULL + (uint64_t)i);
        if (add_chunk_only(h, CHUNK_SIZE, 0) != 0) {
            return &worker_error;
        }
        if (vmm_handle_created(h) != 0) {
            return &worker_error;
        }
        vmm_handle_released(h);
    }
    return NULL;
}

static void *worker_check_type(void *arg) {
    (void)arg;
    int i = 0;
    while (!atomic_load_explicit(&stop_reader, memory_order_acquire)) {
        CUdeviceptr ptr =
            (CUdeviceptr)(0x100000ULL + (uint64_t)(i % ITERATIONS) * CHUNK_SIZE);
        (void)check_memory_type(ptr);
        i++;
    }
    return NULL;
}

int main(void) {
    pthread_t t_alloc, t_vmm, t_reader;
    void *rc_alloc = NULL;
    void *rc_vmm = NULL;

    log_utils_init();
    allocator_init();

    EXPECT(pthread_create(&t_reader, NULL, worker_check_type, NULL) == 0);
    EXPECT(pthread_create(&t_alloc, NULL, worker_alloc_free, NULL) == 0);
    EXPECT(pthread_create(&t_vmm, NULL, worker_vmm_lifecycle, NULL) == 0);

    EXPECT(pthread_join(t_alloc, &rc_alloc) == 0);
    EXPECT(pthread_join(t_vmm, &rc_vmm) == 0);
    atomic_store_explicit(&stop_reader, 1, memory_order_release);
    EXPECT(pthread_join(t_reader, NULL) == 0);

    EXPECT(rc_alloc == NULL);
    EXPECT(rc_vmm == NULL);
    EXPECT(device_overallocated->length == 0);
    EXPECT(device_overallocated->head == NULL);
    EXPECT(device_overallocated->tail == NULL);
    EXPECT(atomic_load_explicit(&removed_bytes, memory_order_relaxed) ==
           (size_t)ITERATIONS * 2 * CHUNK_SIZE);

    if (failures != 0) {
        return 1;
    }
    printf("allocator list locking tests passed\n");
    return 0;
}

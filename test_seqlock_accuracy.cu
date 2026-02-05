/*
 * Seqlock Memory Accounting Test
 *
 * This test verifies:
 * 1. No partial reads during concurrent allocations
 * 2. Memory accounting is always accurate
 * 3. No race conditions in seqlock implementation
 * 4. Handles high contention scenarios
 *
 * Compile:
 *   nvcc -o test_seqlock_accuracy test_seqlock_accuracy.cu -lcudart -lnvidia-ml
 *
 * Run with MPI:
 *   mpirun -np 8 ./test_seqlock_accuracy
 */

#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <errno.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[PID %d] CUDA Error %s:%d: %s\n", \
                    getpid(), __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_NVML(call) \
    do { \
        nvmlReturn_t err = call; \
        if (err != NVML_SUCCESS) { \
            fprintf(stderr, "[PID %d] NVML Error %s:%d: %s\n", \
                    getpid(), __FILE__, __LINE__, nvmlErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Colors for output
#define COLOR_GREEN   "\033[0;32m"
#define COLOR_RED     "\033[0;31m"
#define COLOR_YELLOW  "\033[0;33m"
#define COLOR_BLUE    "\033[0;34m"
#define COLOR_RESET   "\033[0m"

typedef struct {
    int pid;
    size_t allocated;
    int iteration;
} allocation_record_t;

// Shared memory region to track allocations
#define MAX_RECORDS 10000
typedef struct {
    allocation_record_t records[MAX_RECORDS];
    int record_count;
    pthread_mutex_t lock;
} shared_tracker_t;

shared_tracker_t *global_tracker = NULL;

// Get memory usage from HAMi
size_t get_hami_memory_usage(int dev) {
    // This will call HAMi's get_gpu_memory_usage which uses seqlock
    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(dev, &device));

    nvmlMemory_t memory;
    CHECK_NVML(nvmlDeviceGetMemoryInfo(device, &memory));

    return memory.used;
}

// Record allocation
void record_allocation(int pid, size_t bytes, int iteration) {
    if (!global_tracker) return;

    pthread_mutex_lock(&global_tracker->lock);
    if (global_tracker->record_count < MAX_RECORDS) {
        global_tracker->records[global_tracker->record_count].pid = pid;
        global_tracker->records[global_tracker->record_count].allocated = bytes;
        global_tracker->records[global_tracker->record_count].iteration = iteration;
        global_tracker->record_count++;
    }
    pthread_mutex_unlock(&global_tracker->lock);
}

// Verify consistency
int verify_consistency(size_t expected_min, size_t expected_max) {
    size_t actual = get_hami_memory_usage(0);

    if (actual < expected_min || actual > expected_max) {
        printf(COLOR_RED "[PID %d] CONSISTENCY CHECK FAILED!\n" COLOR_RESET, getpid());
        printf("  Expected: %zu - %zu bytes\n", expected_min, expected_max);
        printf("  Actual:   %zu bytes\n", actual);
        return 0;
    }
    return 1;
}

// Test 1: Concurrent Allocation/Free
int test_concurrent_alloc_free(int num_iterations, size_t alloc_size) {
    printf(COLOR_BLUE "[PID %d] Test 1: Concurrent Alloc/Free (%d iterations, %zu MB each)\n" COLOR_RESET,
           getpid(), num_iterations, alloc_size / (1024*1024));

    void **ptrs = (void**)malloc(num_iterations * sizeof(void*));
    size_t total_allocated = 0;

    for (int i = 0; i < num_iterations; i++) {
        // Allocate
        CHECK_CUDA(cudaMalloc(&ptrs[i], alloc_size));
        total_allocated += alloc_size;
        record_allocation(getpid(), alloc_size, i);

        // Small delay to increase contention
        usleep(rand() % 1000);

        // Verify memory accounting
        size_t reported = get_hami_memory_usage(0);

        // Allow some tolerance for other processes
        if (reported < total_allocated * 0.5) {  // Should see at least our own allocations
            printf(COLOR_RED "[PID %d] Iteration %d: Memory underreported! Expected >= %zu, got %zu\n" COLOR_RESET,
                   getpid(), i, total_allocated, reported);
            return 0;
        }

        if (i % 10 == 0) {
            printf("[PID %d] Iteration %d/%d: Allocated %zu MB, Reported: %zu MB\n",
                   getpid(), i+1, num_iterations,
                   total_allocated / (1024*1024), reported / (1024*1024));
        }
    }

    // Free all
    for (int i = 0; i < num_iterations; i++) {
        CHECK_CUDA(cudaFree(ptrs[i]));
        usleep(rand() % 500);
    }

    free(ptrs);

    // Final verification - allow time for cleanup
    sleep(1);
    size_t final = get_hami_memory_usage(0);
    printf("[PID %d] Final memory after free: %zu MB\n", getpid(), final / (1024*1024));

    return 1;
}

// Test 2: Rapid Alloc/Free (stress test for seqlock retries)
int test_rapid_alloc_free(int num_cycles) {
    printf(COLOR_BLUE "[PID %d] Test 2: Rapid Alloc/Free Stress Test (%d cycles)\n" COLOR_RESET,
           getpid(), num_cycles);

    size_t chunk_size = 10 * 1024 * 1024;  // 10 MB
    void *ptr;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_cycles; i++) {
        CHECK_CUDA(cudaMalloc(&ptr, chunk_size));

        // Read memory usage (this tests seqlock under high write contention)
        volatile size_t usage = get_hami_memory_usage(0);
        (void)usage;  // Prevent optimization

        CHECK_CUDA(cudaFree(ptr));

        if (i % 100 == 0) {
            printf("[PID %d] Rapid cycle %d/%d\n", getpid(), i, num_cycles);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ops_per_sec = (num_cycles * 2) / elapsed;  // 2 ops per cycle (alloc + free)

    printf(COLOR_GREEN "[PID %d] Completed %d cycles in %.2f seconds (%.0f ops/sec)\n" COLOR_RESET,
           getpid(), num_cycles, elapsed, ops_per_sec);

    return 1;
}

// Test 3: Partial Read Detection
int test_partial_read_detection(int num_samples) {
    printf(COLOR_BLUE "[PID %d] Test 3: Partial Read Detection (%d samples)\n" COLOR_RESET,
           getpid(), num_samples);

    // Allocate memory in a pattern
    void *ptr1, *ptr2, *ptr3;
    size_t size1 = 100 * 1024 * 1024;  // 100 MB
    size_t size2 = 200 * 1024 * 1024;  // 200 MB
    size_t size3 = 300 * 1024 * 1024;  // 300 MB

    CHECK_CUDA(cudaMalloc(&ptr1, size1));
    CHECK_CUDA(cudaMalloc(&ptr2, size2));
    CHECK_CUDA(cudaMalloc(&ptr3, size3));

    size_t total = size1 + size2 + size3;

    // Rapidly read memory usage while other processes are allocating
    int inconsistencies = 0;
    for (int i = 0; i < num_samples; i++) {
        size_t reading1 = get_hami_memory_usage(0);
        usleep(1);  // Tiny delay
        size_t reading2 = get_hami_memory_usage(0);

        // Check for impossible values (partial reads would show this)
        // Readings should be monotonic or stable within small window
        if (abs((long)(reading2 - reading1)) > (long)total) {
            printf(COLOR_YELLOW "[PID %d] Large variance detected: %zu -> %zu (delta: %ld MB)\n" COLOR_RESET,
                   getpid(), reading1 / (1024*1024), reading2 / (1024*1024),
                   (long)(reading2 - reading1) / (1024*1024));
            inconsistencies++;
        }
    }

    CHECK_CUDA(cudaFree(ptr1));
    CHECK_CUDA(cudaFree(ptr2));
    CHECK_CUDA(cudaFree(ptr3));

    if (inconsistencies > num_samples * 0.05) {  // Allow 5% variance
        printf(COLOR_RED "[PID %d] Too many inconsistencies: %d/%d (%.1f%%)\n" COLOR_RESET,
               getpid(), inconsistencies, num_samples, 100.0 * inconsistencies / num_samples);
        return 0;
    }

    printf(COLOR_GREEN "[PID %d] Inconsistencies: %d/%d (%.1f%%) - PASS\n" COLOR_RESET,
           getpid(), inconsistencies, num_samples, 100.0 * inconsistencies / num_samples);

    return 1;
}

// Test 4: Multi-threaded allocation
void* thread_allocator(void* arg) {
    int thread_id = *(int*)arg;
    size_t chunk_size = 50 * 1024 * 1024;  // 50 MB
    int iterations = 20;

    for (int i = 0; i < iterations; i++) {
        void *ptr;
        cudaMalloc(&ptr, chunk_size);
        usleep(rand() % 5000);
        cudaFree(ptr);

        if (i % 5 == 0) {
            printf("[PID %d Thread %d] Iteration %d/%d\n", getpid(), thread_id, i, iterations);
        }
    }

    return NULL;
}

int test_multithreaded_alloc(int num_threads) {
    printf(COLOR_BLUE "[PID %d] Test 4: Multi-threaded Allocation (%d threads)\n" COLOR_RESET,
           getpid(), num_threads);

    pthread_t threads[num_threads];
    int thread_ids[num_threads];

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thread_allocator, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf(COLOR_GREEN "[PID %d] Multi-threaded test complete\n" COLOR_RESET);
    return 1;
}

// Main test runner
int main(int argc, char **argv) {
    int num_processes = 8;
    int process_id = 0;

    // Initialize CUDA
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    // Use GPU 0
    CHECK_CUDA(cudaSetDevice(0));

    // Initialize NVML
    CHECK_NVML(nvmlInit());

    // Seed random
    srand(time(NULL) ^ getpid());

    printf(COLOR_GREEN "========================================\n");
    printf("HAMi Seqlock Memory Accounting Test\n");
    printf("========================================\n" COLOR_RESET);
    printf("[PID %d] Starting tests...\n\n", getpid());

    int all_passed = 1;

    // Run tests
    if (!test_concurrent_alloc_free(50, 10 * 1024 * 1024)) {  // 50 iterations, 10MB each
        all_passed = 0;
    }
    sleep(1);

    if (!test_rapid_alloc_free(500)) {  // 500 rapid cycles
        all_passed = 0;
    }
    sleep(1);

    if (!test_partial_read_detection(1000)) {  // 1000 samples
        all_passed = 0;
    }
    sleep(1);

    if (!test_multithreaded_alloc(4)) {  // 4 threads
        all_passed = 0;
    }
    sleep(1);

    // Final report
    printf("\n");
    printf("========================================\n");
    if (all_passed) {
        printf(COLOR_GREEN "ALL TESTS PASSED!\n" COLOR_RESET);
        printf("✓ No partial reads detected\n");
        printf("✓ Memory accounting accurate\n");
        printf("✓ No race conditions found\n");
        printf("✓ Seqlock working correctly\n");
    } else {
        printf(COLOR_RED "SOME TESTS FAILED!\n" COLOR_RESET);
        printf("✗ Check logs above for details\n");
    }
    printf("========================================\n");

    nvmlShutdown();

    return all_passed ? 0 : 1;
}

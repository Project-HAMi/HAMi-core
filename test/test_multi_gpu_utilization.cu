/**
 * test_multi_gpu_utilization.cu
 *
 * Test for multi-GPU independent token pool functionality.
 *
 * This test verifies that GPU utilization limiting works independently
 * for each GPU device, which is essential for VLLM TP>1 scenarios.
 *
 * Test Scenarios:
 *   1. Both GPUs run IDENTICAL workload with SAME SM limit
 *   2. If token pools are SHARED , GPUs will block each other
 *   3. If token pools are INDEPENDENT (correct), both complete normally
 *
 * Usage:
 *   rm -f /tmp/cudevshr.cache
 *   export CUDA_DEVICE_SM_LIMIT=25
 *   export GPU_CORE_UTILIZATION_POLICY=FORCE
 *   LD_PRELOAD=./build/libvgpu.so ./build/test/test_multi_gpu_utilization
 *
 * Expected behavior:
 *   - Each GPU should have its own token pool
 *   - Both GPUs should complete in similar time (not blocking each other)
 *   - If GPU1 waits for GPU0, it indicates SHARED token pool
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <time.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Busy kernel that consumes GPU resources
__global__ void busyKernel(double* data, int N, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        double sum = 0.0;
        for (int i = 0; i < iterations; i++) {
            sum += sin(data[tid] * 0.001) * cos(data[tid] * 0.001);
            data[tid] = sum;
        }
    }
}

// Thread argument structure
typedef struct {
    int device_id;
    int num_kernels;
    int iterations;
    double elapsed_time;
    int completed;
} ThreadArg;

// Thread function to run kernels on a specific GPU
void* runOnDevice(void* arg) {
    ThreadArg* ta = (ThreadArg*)arg;
    struct timespec start, end;

    CHECK_CUDA(cudaSetDevice(ta->device_id));

    // Allocate memory
    int N = 1 << 24;  // ~16M elements
    double* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_data, 0, N * sizeof(double)));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Launch multiple kernels
    for (int i = 0; i < ta->num_kernels; i++) {
        busyKernel<<<blocks, threadsPerBlock>>>(d_data, N, ta->iterations);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &end);

    ta->elapsed_time = (end.tv_sec - start.tv_sec) +
                       (end.tv_nsec - start.tv_nsec) / 1e9;
    ta->completed = 1;

    CHECK_CUDA(cudaFree(d_data));

    printf("[GPU %d] Completed %d kernels (%d iterations each) in %.2f seconds\n",
           ta->device_id, ta->num_kernels, ta->iterations, ta->elapsed_time);

    return NULL;
}

int main(int argc, char** argv) {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    printf("========================================\n");
    printf("Multi-GPU Token Pool Isolation Test\n");
    printf("========================================\n");
    printf("Found %d GPU(s)\n", deviceCount);

    if (deviceCount < 2) {
        printf("\n[WARNING] This test requires at least 2 GPUs.\n");
        printf("Current system only has %d GPU(s).\n", deviceCount);
        printf("The test will run in single-GPU mode for basic validation.\n\n");

        // Fall back to single GPU test
        pthread_t thread;
        ThreadArg arg = {0, 5, 2000, 0.0, 0};

        printf("Running single-GPU test...\n");
        pthread_create(&thread, NULL, runOnDevice, &arg);
        pthread_join(thread, NULL);

        printf("\n[RESULT] Single-GPU test completed in %.2f seconds.\n", arg.elapsed_time);
        return 0;
    }

    // Multi-GPU test: IDENTICAL workload on both GPUs
    printf("\nTest: Both GPUs run IDENTICAL workload\n");
    printf("If token pools are SHARED, GPUs will BLOCK each other\n");
    printf("If token pools are INDEPENDENT, both complete normally\n\n");

    pthread_t threads[2];
    ThreadArg args[2];

    // IDENTICAL workload for both GPUs
    int num_kernels = 5;
    int iterations = 2000;

    // GPU0
    args[0].device_id = 0;
    args[0].num_kernels = num_kernels;
    args[0].iterations = iterations;
    args[0].elapsed_time = 0.0;
    args[0].completed = 0;

    // GPU1 - SAME workload as GPU0
    args[1].device_id = 1;
    args[1].num_kernels = num_kernels;
    args[1].iterations = iterations;
    args[1].elapsed_time = 0.0;
    args[1].completed = 0;

    struct timespec total_start, total_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    // Start both threads simultaneously
    pthread_create(&threads[0], NULL, runOnDevice, &args[0]);
    pthread_create(&threads[1], NULL, runOnDevice, &args[1]);

    // Wait for both to complete
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    double total_time = (total_end.tv_sec - total_start.tv_sec) +
                        (total_end.tv_nsec - total_start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("RESULTS:\n");
    printf("========================================\n");
    printf("GPU0: %.2f seconds\n", args[0].elapsed_time);
    printf("GPU1: %.2f seconds\n", args[1].elapsed_time);
    printf("Total: %.2f seconds\n", total_time);
    printf("Time difference: %.2f seconds (%.1f%%)\n",
           fabs(args[0].elapsed_time - args[1].elapsed_time),
           fabs(args[0].elapsed_time - args[1].elapsed_time) /
           ((args[0].elapsed_time + args[1].elapsed_time) / 2) * 100);
    printf("\n");

    // Analysis
    double max_time = args[0].elapsed_time > args[1].elapsed_time ?
                      args[0].elapsed_time : args[1].elapsed_time;
    double min_time = args[0].elapsed_time < args[1].elapsed_time ?
                      args[0].elapsed_time : args[1].elapsed_time;
    double ratio = max_time / min_time;

    if (ratio < 1.2) {
        printf("[PASS] Both GPUs completed in similar time (ratio=%.2fx)\n", ratio);
        printf("       This indicates per-GPU token pools are INDEPENDENT.\n");
        printf("       No blocking between GPUs detected.\n");
    } else {
        printf("[FAIL] Significant time difference (ratio=%.2fx)\n", ratio);
        printf("       One GPU may have been BLOCKED by the other.\n");
        printf("       This could indicate SHARED token pool.\n");
    }

    printf("\n[NOTE] Test prerequisites:\n");
    printf("       rm -f /tmp/cudevshr.cache\n");
    printf("       export CUDA_DEVICE_SM_LIMIT=25\n");
    printf("       export GPU_CORE_UTILIZATION_POLICY=FORCE\n");

    return 0;
}

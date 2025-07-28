#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "test_utils.h"
#include <unistd.h>


__global__ void add(float* a, float* b, float* c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

__global__ void computeKernel(double* data, int N, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        double temp = 0.0;
        temp += sin(data[tid]) * cos(data[tid]);
        data[tid] = temp;
    }
}

int main() {
    float *a, *b, *c;
    CHECK_RUNTIME_API(cudaMalloc(&a, 1024 * sizeof(float)));
    CHECK_RUNTIME_API(cudaMalloc(&b, 1024 * sizeof(float)));
    CHECK_RUNTIME_API(cudaMalloc(&c, 1024 * sizeof(float)));

    add<<<1, 1024>>>(a, b, c);

    int N = 1 << 27; 
    double* d_data;

    cudaMalloc(&d_data, N * sizeof(double));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int iterations = 1000000; 
    int num_launches = 100; 

    for (int i = 0; i < num_launches; ++i) {
        computeKernel<<<blocks, threadsPerBlock>>>(d_data, N, iterations);
        cudaDeviceSynchronize();  
    }

    cudaFree(d_data);

    sleep(100);
    printf("completed");
    return 0;
}

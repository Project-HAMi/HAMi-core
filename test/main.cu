#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    size_t free_mem = 0, total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        printf("cudaMemGetInfo error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Reported Total: %zu MB\n", total_mem / (1024 * 1024));
    printf("Reported Free:  %zu MB\n", free_mem / (1024 * 1024));

    void* dev_ptr = NULL;
    // Changed: Allocate 10MB instead of 2GB to prevent VRAM exhaustion
    err = cudaMalloc(&dev_ptr, 10ULL * 1024 * 1024); 
    if (err == cudaSuccess) {
        printf("[SUCCESS] Allocated 10 MB\n"); // Changed: Updated print statement
        cudaFree(dev_ptr);
    } else {
        printf("[FAILED] Allocation error: %s\n", cudaGetErrorString(err));
    }
    return 0;
}

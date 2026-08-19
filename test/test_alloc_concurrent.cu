#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <cuda_runtime.h>

#define NUM_WORKERS 4
#define ITERATIONS 10

void worker_task(int worker_id) {
    void *ptr = NULL;
    for (int i = 0; i < ITERATIONS; i++) {
        // Allocate 1MB to trigger HAMi-core's memory interception
        cudaError_t err = cudaMalloc(&ptr, 1024 * 1024);
        if (err != cudaSuccess) {
            printf("Worker %d failed allocation: %s\n", worker_id, cudaGetErrorString(err));
            exit(1);
        }
        cudaFree(ptr);
    }
    printf("Worker %d completed successfully.\n", worker_id);
    exit(0);
}

int main() {
    printf("Starting %d concurrent workers...\n", NUM_WORKERS);
    
    for (int i = 0; i < NUM_WORKERS; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            worker_task(i);
        } else if (pid < 0) {
            printf("Fork failed.\n");
            return 1;
        }
    }

    // Parent waits for all children to finish
    int status;
    while (wait(&status) > 0);
    
    printf("All workers finished.\n");
    return 0;
}
// test/bench_lock.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <assert.h>

// Forward declare the lock functions from src/utils.c
extern int try_lock_unified_lock(void);
extern int try_unlock_unified_lock(void);

// A simple function to get time in milliseconds
long long current_timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000LL + te.tv_usec / 1000;
}

void worker(int worker_id, int iterations) {
    for (int i = 0; i < iterations; i++) {
        if (try_lock_unified_lock() != 0) {
            fprintf(stderr, "Worker %d failed to acquire lock\n", worker_id);
            exit(1);
        }
        
        // Critical section: in a real test, you'd write to a shared memory counter here.
        // We do a tiny sleep to simulate the shared memory setup HAMi does during init.
        usleep(500); 

        if (try_unlock_unified_lock() != 0) {
            fprintf(stderr, "Worker %d failed to release lock\n", worker_id);
            exit(1);
        }
    }
    exit(0);
}

int main(int argc, char **argv) {
    int num_workers = 50; // Simulate 50 concurrent CUDA processes
    int iterations = 10;
    
    printf("Starting benchmark: %d workers, %d iterations each...\n", num_workers, iterations);
    long long start_time = current_timestamp();

    for (int i = 0; i < num_workers; i++) {
        if (fork() == 0) {
            worker(i, iterations);
        }
    }

    int status;
    while (wait(&status) > 0) {
        if (WEXITSTATUS(status) != 0) {
            printf("A worker failed!\n");
            return 1;
        }
    }

    long long end_time = current_timestamp();
    printf("All workers finished successfully!\n");
    printf("Total time taken: %lld ms\n", end_time - start_time);
    
    return 0;
}

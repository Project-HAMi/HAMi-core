/*
 * GPU-free regression test for the ctx_activate primary-context flag.
 *
 * cuDevicePrimaryCtxRetain and cuDevicePrimaryCtxRelease_v2 run on any thread
 * and charge context_size to the process once per device.  The flag that gates
 * that accounting has to carry the 0 <-> 1 transition itself, so exactly one
 * thread performs the add and exactly one performs the matching remove.
 *
 * The invariant checked here is that the two stay balanced.  Every add must be
 * paired with a remove, otherwise the process accumulates charges for contexts
 * it never held and starts failing allocations against memory it is not using.
 *
 * With a plain int and a separate test-then-set, four threads drift by
 * thousands of unpaired adds over this many iterations.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define THREAD_COUNT 4
#define ITERATIONS 200000
#define TEST_DEV 0

static _Atomic long adds;
static _Atomic long removes;

static void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        /* Stands in for add_gpu_device_memory_usage(getpid(), dev, context_size, 0). */
        if (ctx_activate_acquire(TEST_DEV)) {
            atomic_fetch_add(&adds, 1);
        }
        /* Stands in for rm_gpu_device_memory_usage(getpid(), dev, context_size, 0). */
        if (ctx_activate_release(TEST_DEV)) {
            atomic_fetch_add(&removes, 1);
        }
    }
    return NULL;
}

int main(void) {
    pthread_t threads[THREAD_COUNT];

    for (int i = 0; i < THREAD_COUNT; i++) {
        if (pthread_create(&threads[i], NULL, worker, NULL) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return 1;
        }
    }
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }

    long a = atomic_load(&adds);
    long r = atomic_load(&removes);
    int failures = 0;

    if (a != r) {
        fprintf(stderr, "unbalanced context accounting: %ld add(s), %ld remove(s), drift %ld\n",
                a, r, a - r);
        failures++;
    }
    if (atomic_load(&ctx_activate[TEST_DEV]) != 0) {
        fprintf(stderr, "ctx_activate[%d] ended set\n", TEST_DEV);
        failures++;
    }
    if (a == 0) {
        fprintf(stderr, "no transitions observed, test did not exercise the flag\n");
        failures++;
    }

    if (failures != 0) {
        return 1;
    }
    printf("ctx_activate race test passed (%ld paired add/remove)\n", a);
    return 0;
}

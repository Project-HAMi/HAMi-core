/*
 * GPU-free regression test for seqlock writer-writer exclusion.
 *
 * get_gpu_memory_usage() reads each process slot under a seqlock.  That
 * protocol is only sound when writers exclude one another: if two writers
 * interleave their sequence increments, the counter can return to an even
 * value while stores are still in flight and a reader will accept a torn
 * snapshot.
 *
 * Two writers hammer the same slot through the public entry points while a
 * reader samples the counter.  The test fails if it ever observes an even
 * sequence number together with a slot total that is not a multiple of the
 * per-write quantum, which is only reachable when a second writer has driven
 * the counter even mid-write.
 *
 * Nothing here calls a CUDA or NVML entry point.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>

/* region_info is static, so reaching the slot to observe the sequence counter
 * means compiling against the translation unit itself.  The CMake target for
 * this test therefore does not list the .c file separately. */
#include "multiprocess/multiprocess_memory_limit.c"  // NOLINT(build/include)

/* The production mapping lives in the NVML-facing utilization watcher, which a
 * GPU-free test must not pull in.  Slot accounting here uses a single device,
 * so the identity mapping is the correct stand-in. */
unsigned int cuda_to_nvml_map(unsigned int cudadev) { return cudadev; }

#define QUANTUM      4096u          /* every write moves total by this much */
#define ITERATIONS   200000
#define DEV          0

static _Atomic int stop_readers;
static _Atomic int readers_ready;
static _Atomic int64_t torn_reads;
static _Atomic int64_t odd_observations;

static void *writer_thread(void *arg) {
    (void)arg;
    for (int64_t i = 0; i < ITERATIONS; i++) {
        add_gpu_device_memory_usage(getpid(), DEV, QUANTUM, 2);
        rm_gpu_device_memory_usage(getpid(), DEV, QUANTUM, 2);
    }
    return NULL;
}

/*
 * Sample the slot the way get_gpu_memory_usage() does.  A snapshot taken
 * while the sequence number is even must be self-consistent, so the running
 * total has to be a whole number of quanta.
 */
static void *reader_thread(void *arg) {
    shrreg_proc_slot_t *slot = region_info.my_slot;
    (void)arg;

    atomic_fetch_add_explicit(&readers_ready, 1, memory_order_release);
    while (!atomic_load_explicit(&stop_readers, memory_order_relaxed)) {
        uint64_t seq1 = atomic_load_explicit(&slot->seqlock, memory_order_acquire);
        if (seq1 & 1) {
            atomic_fetch_add_explicit(&odd_observations, 1, memory_order_relaxed);
            continue;
        }
        uint64_t total = atomic_load_explicit(&slot->used[DEV].total,
                                              memory_order_acquire);
        uint64_t data  = atomic_load_explicit(&slot->used[DEV].data_size,
                                              memory_order_acquire);
        atomic_thread_fence(memory_order_acquire);
        uint64_t seq2 = atomic_load_explicit(&slot->seqlock, memory_order_acquire);

        if (seq1 != seq2)
            continue;                       /* a write landed, resample */

        /* Accepted as a consistent snapshot.  total and data_size move
         * together in one critical section, so they must agree. */
        if (total % QUANTUM != 0 || data % QUANTUM != 0 || total != data)
            atomic_fetch_add_explicit(&torn_reads, 1, memory_order_relaxed);
    }
    return NULL;
}

int main(void) {
    pthread_t w1, w2, r1, r2;

    /* overwrite, so an inherited value cannot change what this test measures */
    if (setenv("CUDA_DEVICE_MEMORY_LIMIT", "0", 1) != 0) {
        perror("FAIL: setenv");
        return 1;
    }
    ensure_initialized();

    if (region_info.my_slot == NULL) {
        fprintf(stderr, "SKIP: no slot allocated for this process\n");
        return 77;                          /* ctest "skipped" convention */
    }

    /* Without this check a failed create leaves an uninitialised pthread_t for
     * join, and the assertions below would still see zero torn reads and report
     * PASS having exercised no concurrency at all. */
    if (pthread_create(&r1, NULL, reader_thread, NULL) != 0 ||
        pthread_create(&r2, NULL, reader_thread, NULL) != 0) {
        fprintf(stderr, "FAIL: pthread_create failed\n");
        return 1;
    }
    /* Hold the writers until both readers are sampling. Otherwise the scheduler
     * can run the writers to completion first and the test passes without ever
     * exercising the torn-snapshot path. */
    while (atomic_load_explicit(&readers_ready, memory_order_acquire) < 2)
        sched_yield();

    if (pthread_create(&w1, NULL, writer_thread, NULL) != 0 ||
        pthread_create(&w2, NULL, writer_thread, NULL) != 0) {
        fprintf(stderr, "FAIL: pthread_create failed\n");
        return 1;
    }

    pthread_join(w1, NULL);
    pthread_join(w2, NULL);
    atomic_store_explicit(&stop_readers, 1, memory_order_relaxed);
    pthread_join(r1, NULL);
    pthread_join(r2, NULL);

    int64_t torn = atomic_load(&torn_reads);
    int64_t odd  = atomic_load(&odd_observations);
    shrreg_proc_slot_t *slot = region_info.my_slot;
    uint64_t final_seq   = atomic_load(&slot->seqlock);
    uint64_t final_total = atomic_load(&slot->used[DEV].total);

    printf("writers=2 readers=2 iterations=%d\n", ITERATIONS);
    printf("odd sequence observations : %" PRId64 "\n", odd);
    printf("torn reads                : %" PRId64 "\n", torn);
    printf("final seqlock             : %" PRIu64 " (%s)\n", final_seq,
           (final_seq & 1) ? "ODD - a writer did not release" : "even");
    printf("final total               : %" PRIu64 " (expected 0)\n", final_total);

    if (torn != 0) {
        fprintf(stderr, "FAIL: %" PRId64 " torn reads accepted under an even sequence\n",
                torn);
        return 1;
    }
    /* two writers, ITERATIONS rounds, an add and a remove each, two
     * transitions per critical section */
    const uint64_t expected_seq = 2ULL * ITERATIONS * 2ULL * 2ULL;
    if (final_seq != expected_seq) {
        fprintf(stderr, "FAIL: sequence is %" PRIu64 ", expected %" PRIu64 "\n",
                final_seq, expected_seq);
        return 1;
    }
    if (final_total != 0) {
        fprintf(stderr, "FAIL: add/rm pairs did not balance, lost updates\n");
        return 1;
    }
    /* Zero means no reader ever sampled while a write was in progress, so the
     * run proves nothing about tearing whatever the other counters say. */
    if (odd == 0) {
        fprintf(stderr, "FAIL: no reader observed an active writer, "
                        "readers and writers did not overlap\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}

/*
 * Cross-process regression test for seqlock writer exclusion.
 *
 * test_seqlock_writer_exclusion covers two threads in one process, which only
 * exercises the pid == getpid() fast path on the cached slot pointer.  The
 * reason the exclusion has to live in the shared region, rather than in a
 * pthread mutex, is the slow path: add_gpu_device_memory_usage and
 * rm_gpu_device_memory_usage look a slot up by pid and write another process's
 * slot.  A process-local mutex cannot exclude a writer in a different process.
 *
 * Every participant here writes every slot by pid, so all but one of its writes
 * take the slow path, while sampling every slot the way get_gpu_memory_usage
 * does.  A snapshot accepted under an even sequence must have total and
 * data_size in agreement, since both move inside one critical section.
 *
 * Nothing here calls a CUDA or NVML entry point.
 */
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <inttypes.h>
#include <sched.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.c"  // NOLINT(build/include)

/* both live in the NVML-facing utilization watcher, which a GPU-free test must
 * not pull in */
unsigned int cuda_to_nvml_map(unsigned int cudadev) { return cudadev; }
int setspec(void) { return 0; }

#define PARTICIPANTS 4                  /* the parent plus three children */
#define QUANTUM      4096u
#define ITERATIONS   20000
#define DEV          0

/* placed in an anonymous shared mapping before the fork, so every participant
 * sees the same object */
typedef struct {
    _Atomic int32_t pid[PARTICIPANTS];
    _Atomic int64_t torn;
    _Atomic int64_t overlaps;
    _Atomic int     ready;
} board_t;

static board_t *board;

static void publish(int index) {
    atomic_store_explicit(&board->pid[index], getpid(), memory_order_release);
}

static void wait_for_everyone(void) {
    for (;;) {
        int seen = 0;
        for (int i = 0; i < PARTICIPANTS; i++)
            if (atomic_load_explicit(&board->pid[i], memory_order_acquire) > 0)
                seen++;
        if (seen == PARTICIPANTS)
            return;
        sched_yield();
    }
}

/* Sample every registered slot the way get_gpu_memory_usage does. */
static void sample_all(void) {
    shared_region_t *region = region_info.shared_region;
    int proc_num = atomic_load_explicit(&region->proc_num, memory_order_acquire);
    for (int i = 0; i < proc_num; i++) {
        shrreg_proc_slot_t *slot = &region->procs[i];
        uint64_t s1 = atomic_load_explicit(&slot->seqlock, memory_order_acquire);
        if (s1 & 1) {
            atomic_fetch_add_explicit(&board->overlaps, 1, memory_order_relaxed);
            continue;
        }
        uint64_t total = atomic_load_explicit(&slot->used[DEV].total,
                                              memory_order_acquire);
        uint64_t data = atomic_load_explicit(&slot->used[DEV].data_size,
                                             memory_order_acquire);
        atomic_thread_fence(memory_order_acquire);
        uint64_t s2 = atomic_load_explicit(&slot->seqlock, memory_order_acquire);
        if (s1 != s2)
            continue;
        if (total != data)
            atomic_fetch_add_explicit(&board->torn, 1, memory_order_relaxed);
    }
}

/* Write every slot by pid.  Only the entry matching getpid() takes the fast
 * path; the rest go through the slow path lookup, which is the case this test
 * exists for. */
static void hammer(void) {
    for (int n = 0; n < ITERATIONS; n++) {
        for (int i = 0; i < PARTICIPANTS; i++) {
            int32_t target = atomic_load_explicit(&board->pid[i],
                                                  memory_order_relaxed);
            if (target <= 0)
                continue;
            add_gpu_device_memory_usage(target, DEV, QUANTUM, 2);
            rm_gpu_device_memory_usage(target, DEV, QUANTUM, 2);
        }
        if ((n & 0x3f) == 0)
            sample_all();
    }
}

int main(void) {
    board = mmap(NULL, sizeof(board_t), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (board == MAP_FAILED) {
        perror("FAIL: mmap");
        return 1;
    }
    if (setenv("CUDA_DEVICE_MEMORY_LIMIT", "0", 1) != 0) {
        perror("FAIL: setenv");
        return 1;
    }

    /* Fork before touching the region.  Initialising first would leave every
     * child sharing the parent's cached slot instead of registering its own. */
    pid_t kids[PARTICIPANTS - 1];
    for (int k = 0; k < PARTICIPANTS - 1; k++) {
        kids[k] = fork();
        if (kids[k] < 0) {
            perror("FAIL: fork");
            return 1;
        }
        if (kids[k] == 0) {
            ensure_initialized();
            if (region_info.my_slot == NULL)
                _exit(77);
            publish(k + 1);
            wait_for_everyone();
            hammer();
            _exit(0);
        }
    }

    ensure_initialized();
    if (region_info.my_slot == NULL) {
        fprintf(stderr, "SKIP: no slot allocated\n");
        for (int k = 0; k < PARTICIPANTS - 1; k++) kill(kids[k], SIGKILL);
        return 77;
    }
    publish(0);
    wait_for_everyone();
    hammer();

    int skipped = 0;
    for (int k = 0; k < PARTICIPANTS - 1; k++) {
        int st = 0;
        waitpid(kids[k], &st, 0);
        if (WIFEXITED(st) && WEXITSTATUS(st) == 77) {
            skipped = 1;
        } else if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            fprintf(stderr, "FAIL: child %d did not exit cleanly (status %d)\n",
                    kids[k], st);
            return 1;
        }
    }
    if (skipped) {
        fprintf(stderr, "SKIP: a participant could not register a slot\n");
        return 77;
    }

    int64_t torn = atomic_load(&board->torn);
    int64_t overlaps = atomic_load(&board->overlaps);
    printf("participants=%d iterations=%d\n", PARTICIPANTS, ITERATIONS);
    printf("writes in progress observed : %" PRId64 "\n", overlaps);
    printf("torn snapshots              : %" PRId64 "\n", torn);

    shared_region_t *region = region_info.shared_region;
    int proc_num = atomic_load_explicit(&region->proc_num, memory_order_acquire);
    for (int i = 0; i < proc_num; i++) {
        uint64_t seq = atomic_load(&region->procs[i].seqlock);
        uint64_t total = atomic_load(&region->procs[i].used[DEV].total);
        if (seq & 1) {
            fprintf(stderr, "FAIL: slot %d left odd, a writer did not release\n", i);
            return 1;
        }
        if (total != 0) {
            fprintf(stderr, "FAIL: slot %d total is %" PRIu64 ", add/rm did not "
                            "balance\n", i, total);
            return 1;
        }
    }
    if (torn != 0) {
        fprintf(stderr, "FAIL: %" PRId64 " torn snapshots across processes\n", torn);
        return 1;
    }
    if (overlaps == 0) {
        fprintf(stderr, "FAIL: never sampled during a write, so the run proves "
                        "nothing\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}

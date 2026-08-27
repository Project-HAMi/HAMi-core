/*
 * Cross-process counterpart to test_seqlock_writer_exclusion, which only covers
 * the pid == getpid() fast path.  Every participant writes every slot by pid, so
 * most writes take the slow path that reaches another process's slot: the case a
 * process-local mutex could not have covered.
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

/* both live in the NVML-facing watcher, which this must not pull in */
unsigned int cuda_to_nvml_map(unsigned int cudadev) { return cudadev; }
int setspec(void) { return 0; }

#define PARTICIPANTS 4                  /* the parent plus three children */
#define QUANTUM      4096u
#define ITERATIONS   20000
#define DEV          0

/* mapped shared before the fork so every participant sees one copy */
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

/* only the getpid() entry takes the fast path; the rest exercise the slow one */
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

    /* fork first: initialising would leave children on the parent's slot */
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

/*
 * bench_init_contention.c
 *
 * Benchmark for HAMi issue #1662: initialization lock contention.
 *
 * This program measures how long HAMi-core initialization takes when many
 * processes initialize CUDA at the same time. It does not change any
 * library behavior. It is a plain consumer of the public CUDA driver API,
 * loaded together with libvgpu.so through LD_PRELOAD, exactly like a real
 * workload process.
 *
 * What it measures:
 *   - per-process cuInit() latency
 *   - total wall time for all processes to finish init
 *   - CPU time and time spent blocked (wall - cpu)
 *   - voluntary context switches (a proxy for blocking on the shared
 *     postInit semaphore)
 *
 * Why cuInit(): in HAMi-core, cuInit() ends by calling ensure_post_init(),
 * which runs postInit() once per process. postInit() takes the shared
 * memory semaphore sem_postinit (lock_postinit) and calls set_task_pid()
 * before releasing it. set_task_pid() runs nvmlInit(), scans running
 * processes twice, and retains a primary CUDA context. Because the
 * semaphore serializes set_task_pid() across every process on the node,
 * timing cuInit() captures the contention this issue is about.
 *
 * Build: this file is picked up automatically by test/CMakeLists.txt.
 *   ./build.sh
 *
 * Run (needs an NVIDIA GPU and the built library):
 *   LD_PRELOAD=./build/libvgpu.so CUDA_VISIBLE_DEVICES=0 \
 *       ./build/test/benchmark/bench_init_contention 128
 *
 * The first argument is the number of processes. The optional second
 * argument is a label printed in the CSV line, useful when sweeping the
 * process count from a shell loop.
 *
 * This benchmark targets Linux, which is the only platform HAMi-core runs
 * on. It relies on fork(), an anonymous shared mapping, a process-shared
 * pthread barrier, getrusage(), and clock_gettime().
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>

#define DEFAULT_PROC_NUM 64
#define MAX_PROC_NUM 4096

typedef struct {
    int valid;          /* 1 if the child filled this record */
    int cu_result;      /* CUresult returned by cuInit()      */
    pid_t pid;          /* child pid, for debugging           */
    double latency_ms;  /* cuInit() wall latency              */
    double cpu_ms;      /* user + system CPU used during init */
    long vol_ctx;       /* voluntary context switches         */
    long invol_ctx;     /* involuntary context switches       */
} child_result_t;

typedef struct {
    pthread_barrier_t barrier;
    child_result_t results[]; /* flexible array, one slot per process */
} bench_shared_t;

static double timespec_diff_ms(const struct timespec *start, const struct timespec *end) {
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
           (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
}

static double timeval_ms(const struct timeval *tv) {
    return (double)tv->tv_sec * 1000.0 + (double)tv->tv_usec / 1000.0;
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Nearest-rank percentile over a sorted array. */
static double percentile(const double *sorted, int n, double p) {
    if (n <= 0) return 0.0;
    int rank = (int)(p / 100.0 * (double)n);
    if (rank >= n) rank = n - 1;
    if (rank < 0) rank = 0;
    return sorted[rank];
}

/* One worker: start together on the barrier, then time cuInit(). */
static void run_child(bench_shared_t *shared, int index) {
    struct rusage ru_start, ru_end;
    struct timespec t0, t1;
    child_result_t *r = &shared->results[index];

    r->pid = getpid();

    /* Release together so the processes contend, matching the real case
     * where many pods start at once. */
    pthread_barrier_wait(&shared->barrier);

    getrusage(RUSAGE_SELF, &ru_start);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    CUresult res = cuInit(0);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    getrusage(RUSAGE_SELF, &ru_end);

    r->cu_result = (int)res;
    r->latency_ms = timespec_diff_ms(&t0, &t1);
    r->cpu_ms = (timeval_ms(&ru_end.ru_utime) - timeval_ms(&ru_start.ru_utime)) +
                (timeval_ms(&ru_end.ru_stime) - timeval_ms(&ru_start.ru_stime));
    r->vol_ctx = ru_end.ru_nvcsw - ru_start.ru_nvcsw;
    r->invol_ctx = ru_end.ru_nivcsw - ru_start.ru_nivcsw;
    r->valid = 1;

    /* _exit() so the child does not flush the parent's stdio buffers. */
    _exit(0);
}

static void report(const char *label, int proc_num, double wall_ms,
                   const child_result_t *results) {
    double *lat = malloc(sizeof(double) * (size_t)proc_num);
    if (lat == NULL) {
        fprintf(stderr, "report: out of memory\n");
        return;
    }

    int ok = 0, failed = 0, missing = 0;
    double cpu_sum = 0.0, vol_sum = 0.0, invol_sum = 0.0;
    for (int i = 0; i < proc_num; i++) {
        if (!results[i].valid) {
            missing++;
            continue;
        }
        if (results[i].cu_result != CUDA_SUCCESS) {
            failed++;
            continue;
        }
        lat[ok] = results[i].latency_ms;
        cpu_sum += results[i].cpu_ms;
        vol_sum += (double)results[i].vol_ctx;
        invol_sum += (double)results[i].invol_ctx;
        ok++;
    }

    printf("\n=== HAMi-core init contention benchmark (issue #1662) ===\n");
    printf("processes launched : %d\n", proc_num);
    printf("cuInit succeeded   : %d\n", ok);
    printf("cuInit failed      : %d\n", failed);
    printf("no result recorded : %d\n", missing);
    printf("total wall time    : %.3f ms\n", wall_ms);

    if (ok > 0) {
        qsort(lat, (size_t)ok, sizeof(double), compare_double);
        double sum = 0.0;
        for (int i = 0; i < ok; i++) sum += lat[i];
        double mean = sum / (double)ok;
        double mean_cpu = cpu_sum / (double)ok;
        double mean_wait = mean - mean_cpu; /* time blocked, not on CPU */

        printf("per-process cuInit() latency (ms):\n");
        printf("  min  : %.3f\n", lat[0]);
        printf("  p50  : %.3f\n", percentile(lat, ok, 50.0));
        printf("  mean : %.3f\n", mean);
        printf("  p95  : %.3f\n", percentile(lat, ok, 95.0));
        printf("  p99  : %.3f\n", percentile(lat, ok, 99.0));
        printf("  max  : %.3f\n", lat[ok - 1]);
        printf("mean CPU per process   : %.3f ms\n", mean_cpu);
        printf("mean blocked per proc  : %.3f ms\n", mean_wait);
        printf("mean vol ctx switches  : %.1f\n", vol_sum / (double)ok);
        printf("mean invol ctx switches: %.1f\n", invol_sum / (double)ok);

        /* Machine-readable line for sweeping process count. */
        printf("CSV,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
               label, proc_num, ok, failed, wall_ms,
               lat[0], percentile(lat, ok, 50.0), mean,
               percentile(lat, ok, 95.0), percentile(lat, ok, 99.0),
               lat[ok - 1], mean_cpu, mean_wait);
    } else {
        printf("no successful cuInit(); check that an NVIDIA GPU is present and "
               "LD_PRELOAD points at the built libvgpu.so\n");
    }

    free(lat);
}

int main(int argc, char **argv) {
    int proc_num = DEFAULT_PROC_NUM;
    const char *label = "run";

    if (argc >= 2) {
        proc_num = atoi(argv[1]);
    }
    if (argc >= 3) {
        label = argv[2];
    }
    if (proc_num < 1 || proc_num > MAX_PROC_NUM) {
        fprintf(stderr, "process count must be between 1 and %d\n", MAX_PROC_NUM);
        return 1;
    }

    size_t shared_size = sizeof(bench_shared_t) +
                         sizeof(child_result_t) * (size_t)proc_num;
    bench_shared_t *shared = mmap(NULL, shared_size, PROT_READ | PROT_WRITE,
                                  MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return 1;
    }
    memset(shared, 0, shared_size);

    pthread_barrierattr_t battr;
    if (pthread_barrierattr_init(&battr) != 0 ||
        pthread_barrierattr_setpshared(&battr, PTHREAD_PROCESS_SHARED) != 0 ||
        pthread_barrier_init(&shared->barrier, &battr, (unsigned)(proc_num + 1)) != 0) {
        fprintf(stderr, "failed to init process-shared barrier\n");
        munmap(shared, shared_size);
        return 1;
    }
    pthread_barrierattr_destroy(&battr);

    pid_t *pids = malloc(sizeof(pid_t) * (size_t)proc_num);
    if (pids == NULL) {
        fprintf(stderr, "out of memory allocating pid table\n");
        pthread_barrier_destroy(&shared->barrier);
        munmap(shared, shared_size);
        return 1;
    }

    int launched = 0;
    for (int i = 0; i < proc_num; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "fork failed at %d: %s\n", i, strerror(errno));
            break;
        }
        if (pid == 0) {
            run_child(shared, i);
            /* not reached */
        }
        pids[launched] = pid;
        launched++;
    }

    if (launched != proc_num) {
        /* Some forks failed. The children that did start are already blocked
         * in pthread_barrier_wait (the barrier needs proc_num + 1 waiters that
         * will never all arrive) and have not run any CUDA yet. A plain wait()
         * here would block forever, so signal them to exit and then reap. */
        fprintf(stderr, "only %d of %d children launched; aborting\n",
                launched, proc_num);
        for (int i = 0; i < launched; i++) kill(pids[i], SIGKILL);
        for (int i = 0; i < launched; i++) waitpid(pids[i], NULL, 0);
        free(pids);
        pthread_barrier_destroy(&shared->barrier);
        munmap(shared, shared_size);
        return 1;
    }

    /* Release all children together and time from here. */
    pthread_barrier_wait(&shared->barrier);
    struct timespec wall_start, wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    for (int i = 0; i < proc_num; i++) {
        wait(NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &wall_end);

    report(label, proc_num, timespec_diff_ms(&wall_start, &wall_end), shared->results);

    free(pids);
    pthread_barrier_destroy(&shared->barrier);
    munmap(shared, shared_size);
    return 0;
}

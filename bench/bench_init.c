/*
Copyright 2024 The HAMi Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
 * bench_init - concurrent CUDA initialization benchmark for HAMi-core.
 *
 * Measures per-process cuInit() latency when N processes initialize at the
 * same instant, which is the scenario described in Project-HAMi/HAMi#1662.
 *
 * Design notes:
 *   - Workers are fork+exec'd, not just forked, so each is a genuinely fresh
 *     process image the way a real pod's child processes are. libvgpu.so is
 *     interposed via LD_PRELOAD and so is inherited across the exec.
 *   - The parent never touches CUDA. Any CUDA state in the parent would be
 *     inherited and would invalidate the host-PID detection being measured.
 *   - Release is broadcast by closing the write end of a pipe. Every worker
 *     is blocked in read(2), so they wake together without spinning, which
 *     keeps the CPU idle during the barrier and avoids polluting the numbers.
 *
 * Usage:
 *   bench_init <n>            run n concurrent workers, print JSON to stdout
 *   bench_init --worker <slot> <readfd>   internal, not for direct use
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <stdatomic.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <cuda.h>

#define MAX_WORKERS 512

/* One slot per worker, in shared memory so the parent can read results back. */
typedef struct {
    double   cuinit_ms;    /* time spent inside cuInit() */
    double   start_off_ms; /* when this worker entered cuInit, relative to release */
    int      rc;           /* CUresult, 0 == CUDA_SUCCESS */
    int      pid;
} slot_t;

typedef struct {
    atomic_int ready;      /* workers that have reached the barrier */
    slot_t     slots[MAX_WORKERS];
} shared_t;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Monotonic clock is shared across processes, so release time is comparable. */
static int run_worker(shared_t *sh, int slot, int readfd) {
    char buf[1];

    atomic_fetch_add(&sh->ready, 1);

    /* Blocks until the parent closes the write end. Returns 0 on EOF. */
    while (read(readfd, buf, 1) < 0 && errno == EINTR) {
    }

    double t0 = now_ms();
    CUresult r = cuInit(0);
    double t1 = now_ms();

    sh->slots[slot].cuinit_ms    = t1 - t0;
    sh->slots[slot].start_off_ms = t0;
    sh->slots[slot].rc           = (int)r;
    sh->slots[slot].pid          = getpid();
    return r == CUDA_SUCCESS ? 0 : 1;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Nearest-rank percentile. With the small N values used here this is more
 * honest than interpolating between samples we did not observe. */
static double pct(double *sorted, int n, double p) {
    if (n <= 0) return 0.0;
    int idx = (int)ceil(p / 100.0 * n) - 1;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--worker") == 0) {
        if (argc != 4) {
            fprintf(stderr, "bad --worker invocation\n");
            return 2;
        }
        int slot   = atoi(argv[2]);
        int readfd = atoi(argv[3]);
        int shmfd  = atoi(getenv("BENCH_SHM_FD"));
        shared_t *sh = mmap(NULL, sizeof(shared_t), PROT_READ | PROT_WRITE,
                            MAP_SHARED, shmfd, 0);
        if (sh == MAP_FAILED) {
            perror("mmap(worker)");
            return 2;
        }
        return run_worker(sh, slot, readfd);
    }

    if (argc != 2) {
        fprintf(stderr, "usage: %s <n_workers>\n", argv[0]);
        return 2;
    }
    int n = atoi(argv[1]);
    if (n < 1 || n > MAX_WORKERS) {
        fprintf(stderr, "n must be 1..%d\n", MAX_WORKERS);
        return 2;
    }

    /* Anonymous shared file, inherited by the workers across exec. */
    int shmfd = memfd_create("bench_shm", 0);
    if (shmfd < 0) {
        perror("memfd_create");
        return 1;
    }
    if (ftruncate(shmfd, sizeof(shared_t)) != 0) {
        perror("ftruncate");
        return 1;
    }

    shared_t *sh = mmap(NULL, sizeof(shared_t), PROT_READ | PROT_WRITE,
                        MAP_SHARED, shmfd, 0);
    if (sh == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    memset(sh, 0, sizeof(*sh));
    atomic_store(&sh->ready, 0);
    /* rc defaults to CUDA_SUCCESS (0) after memset, which would make a worker
     * that dies before writing its result look like a 0ms success. Mark every
     * slot as failed up front; a worker overwrites this only once it finishes. */
    for (int i = 0; i < n; i++) {
        sh->slots[i].rc = -1;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        perror("pipe");
        return 1;
    }

    /* Both fds must survive exec, so clear FD_CLOEXEC. */
    fcntl(pipefd[0], F_SETFD, 0);
    fcntl(shmfd,     F_SETFD, 0);

    char shmfd_s[16];
    snprintf(shmfd_s, sizeof shmfd_s, "%d", shmfd);
    setenv("BENCH_SHM_FD", shmfd_s, 1);

    char self[PATH_MAX];
    ssize_t sl = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (sl < 0) {
        perror("readlink");
        return 1;
    }
    self[sl] = '\0';

    pid_t pids[MAX_WORKERS];
    for (int i = 0; i < n; i++) {
        pid_t p = fork();
        if (p < 0) {
            perror("fork");
            return 1;
        }
        if (p == 0) {
            close(pipefd[1]);
            char slot_s[16], rfd_s[16];
            snprintf(slot_s, sizeof slot_s, "%d", i);
            snprintf(rfd_s,  sizeof rfd_s,  "%d", pipefd[0]);
            execl(self, self, "--worker", slot_s, rfd_s, (char *)NULL);
            perror("execl");
            _exit(127);
        }
        pids[i] = p;
    }
    close(pipefd[0]);

    /* Wait for every worker to reach the barrier before releasing any, but
     * don't spin forever if a worker dies before it gets there (exec/loader
     * failure, killed, etc.). Reap early exits as they happen so a dead
     * worker can't stall the barrier, and give up after a generous timeout. */
    int reaped[MAX_WORKERS];
    memset(reaped, 0, sizeof(reaped));
    int early_exits = 0;
    double ready_deadline = now_ms() + 30000.0; /* 30s */

    for (;;) {
        int ready_n = atomic_load(&sh->ready);
        if (ready_n + early_exits >= n) break;
        if (now_ms() > ready_deadline) {
            fprintf(stderr,
                    "bench_init: timed out waiting for workers to reach barrier "
                    "(%d/%d ready); aborting\n", ready_n, n);
            for (int i = 0; i < n; i++) {
                if (!reaped[i]) kill(pids[i], SIGKILL);
            }
            for (int i = 0; i < n; i++) {
                if (!reaped[i]) waitpid(pids[i], NULL, 0);
            }
            return 1;
        }
        for (int i = 0; i < n; i++) {
            if (reaped[i]) continue;
            int st = 0;
            if (waitpid(pids[i], &st, WNOHANG) == pids[i]) {
                reaped[i] = 1;
                early_exits++;
            }
        }
        usleep(200);
    }

    double t_release = now_ms();
    close(pipefd[1]);                 /* broadcast: all workers wake here */

    int failed = early_exits;
    for (int i = 0; i < n; i++) {
        if (reaped[i]) continue;
        int st = 0;
        waitpid(pids[i], &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) failed++;
    }
    double wall_ms = now_ms() - t_release;

    double lat[MAX_WORKERS];
    int ok = 0;
    double sum = 0.0, max_end = 0.0;
    for (int i = 0; i < n; i++) {
        if (sh->slots[i].rc != 0) continue;
        lat[ok++] = sh->slots[i].cuinit_ms;
        sum += sh->slots[i].cuinit_ms;
        double end = (sh->slots[i].start_off_ms - t_release) + sh->slots[i].cuinit_ms;
        if (end > max_end) max_end = end;
    }
    qsort(lat, ok, sizeof(double), cmp_double);

    printf("{\"n\":%d,\"ok\":%d,\"failed\":%d,"
           "\"wall_ms\":%.2f,\"span_ms\":%.2f,\"mean_ms\":%.2f,"
           "\"p50_ms\":%.2f,\"p95_ms\":%.2f,\"p99_ms\":%.2f,"
           "\"min_ms\":%.2f,\"max_ms\":%.2f}\n",
           n, ok, failed, wall_ms, max_end,
           ok ? sum / ok : 0.0,
           pct(lat, ok, 50), pct(lat, ok, 95), pct(lat, ok, 99),
           ok ? lat[0] : 0.0, ok ? lat[ok - 1] : 0.0);
    return failed ? 1 : 0;
}

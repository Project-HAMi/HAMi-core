/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <nvml.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "../src/include/hostpid_fallback_lock.h"
#include "../src/include/nvml_override.h"
#include "../src/include/nvml_prefix.h"

#define MAX_WORKERS 16
#define POSTINIT_LOCK_OFFSET ((off_t)UINT64_C(0x40000000))
#define WAIT_TIMEOUT_MS 5000

enum classification {
    CLASS_UNSET = 0,
    CLASS_OWN_PID = 1,
    CLASS_PEER_PID = 2,
    CLASS_FAILURE = 3,
    CLASS_TIMEOUT = 4,
};

struct worker_result {
    pid_t expected_pid;
    pid_t observed_pid;
    int cache_index;
    int device_index;
    int classification;
    int64_t lock_wait_us;
    int64_t discovery_us;
};

struct shared_state {
    _Atomic int start;
    _Atomic int before_ready;
    _Atomic int registered_ready;
    _Atomic int after_ready;
    _Atomic int active[MAX_WORKERS];
    struct worker_result results[MAX_WORKERS];
};

FILE *fp1;
int g_log_level;

int getextrapid(unsigned int previous, unsigned int current,
                nvmlProcessInfo_t1 *before, nvmlProcessInfo_t1 *after);

static int64_t monotonic_us(void) {
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1;
    }
    return (int64_t)now.tv_sec * INT64_C(1000000) +
           now.tv_nsec / INT64_C(1000);
}

static int wait_for_value(_Atomic int *value, int expected) {
    int64_t deadline = monotonic_us() +
                       (int64_t)WAIT_TIMEOUT_MS * INT64_C(1000);

    while (atomic_load_explicit(value, memory_order_acquire) < expected) {
        struct timespec delay = {.tv_sec = 0, .tv_nsec = 1000000L};

        if (monotonic_us() >= deadline) {
            errno = ETIMEDOUT;
            return -1;
        }
        nanosleep(&delay, NULL);
    }
    return 0;
}

static int cache_lock(int fd, int16_t type) {
    struct flock lock = {
        .l_type = type,
        .l_whence = SEEK_SET,
        .l_start = POSTINIT_LOCK_OFFSET,
        .l_len = 1,
    };

    for (;;) {
        if (fcntl(fd, type == F_UNLCK ? F_SETLK : F_SETLKW, &lock) == 0) {
            return 0;
        }
        if (errno != EINTR) {
            return -1;
        }
    }
}

static unsigned int snapshot_device(const struct shared_state *state,
                                    int workers, int devices, int device,
                                    nvmlProcessInfo_t1 *snapshot) {
    unsigned int count = 0;
    int index;

    for (index = 0; index < workers; index++) {
        int pid;

        if (index % devices != device) {
            continue;
        }
        pid = atomic_load_explicit(&state->active[index],
                                   memory_order_acquire);
        if (pid > 0) {
            snapshot[count].pid = (unsigned int)pid;
            snapshot[count].usedGpuMemory = 0;
            count++;
        }
    }
    return count;
}

static int worker_main(struct shared_state *state, const char *lock_directory,
                       const char *cache_path, int worker, int workers,
                       int devices, int cache_index, int use_global_lock,
                       int force_overlap) {
    nvmlProcessInfo_t1 before[MAX_WORKERS] = {{0}};
    nvmlProcessInfo_t1 after[MAX_WORKERS] = {{0}};
    struct worker_result *result = &state->results[worker];
    unsigned int before_count;
    unsigned int after_count;
    int64_t lock_start;
    int64_t discovery_start;
    int cache_fd;

    result->expected_pid = getpid();
    result->cache_index = cache_index;
    result->device_index = worker % devices;
    result->classification = CLASS_FAILURE;

    if (wait_for_value(&state->start, 1) != 0) {
        result->classification = CLASS_TIMEOUT;
        return 2;
    }
    cache_fd = open(cache_path, O_RDWR | O_CLOEXEC);
    if (cache_fd < 0) {
        return 3;
    }

    lock_start = monotonic_us();
    if (use_global_lock &&
        hostpid_fallback_lock_acquire_at(lock_directory, getuid(),
                                         WAIT_TIMEOUT_MS) != 0) {
        result->classification = errno == ETIMEDOUT ? CLASS_TIMEOUT
                                                     : CLASS_FAILURE;
        close(cache_fd);
        return 4;
    }
    if (cache_lock(cache_fd, F_WRLCK) != 0) {
        if (use_global_lock) {
            hostpid_fallback_lock_release();
        }
        close(cache_fd);
        return 5;
    }
    result->lock_wait_us = monotonic_us() - lock_start;
    discovery_start = monotonic_us();

    before_count = snapshot_device(state, workers, devices,
                                   result->device_index, before);
    if (force_overlap) {
        atomic_fetch_add_explicit(&state->before_ready, 1,
                                  memory_order_acq_rel);
        if (wait_for_value(&state->before_ready, workers) != 0) {
            result->classification = CLASS_TIMEOUT;
            goto out;
        }
    }

    atomic_store_explicit(&state->active[worker], (int)getpid(),
                          memory_order_release);
    if (force_overlap) {
        atomic_fetch_add_explicit(&state->registered_ready, 1,
                                  memory_order_acq_rel);
        if (wait_for_value(&state->registered_ready, workers) != 0) {
            result->classification = CLASS_TIMEOUT;
            goto out;
        }
    } else {
        struct timespec overlap = {.tv_sec = 0, .tv_nsec = 20000000L};

        nanosleep(&overlap, NULL);
    }

    after_count = snapshot_device(state, workers, devices,
                                  result->device_index, after);
    result->observed_pid = (pid_t)getextrapid(before_count, after_count,
                                              before, after);
    if (force_overlap) {
        atomic_fetch_add_explicit(&state->after_ready, 1,
                                  memory_order_acq_rel);
        if (wait_for_value(&state->after_ready, workers) != 0) {
            result->classification = CLASS_TIMEOUT;
            goto out;
        }
    }
    if (result->observed_pid == result->expected_pid) {
        result->classification = CLASS_OWN_PID;
    } else if (result->observed_pid > 0) {
        result->classification = CLASS_PEER_PID;
    } else {
        result->classification = CLASS_FAILURE;
    }
    result->discovery_us = monotonic_us() - discovery_start;

out:
    atomic_store_explicit(&state->active[worker], 0, memory_order_release);
    cache_lock(cache_fd, F_UNLCK);
    if (use_global_lock) {
        hostpid_fallback_lock_release();
    }
    close(cache_fd);
    return result->classification == CLASS_OWN_PID ? 0 : 1;
}

static const char *classification_name(int classification) {
    switch (classification) {
        case CLASS_OWN_PID:
            return "own_pid";
        case CLASS_PEER_PID:
            return "peer_pid";
        case CLASS_TIMEOUT:
            return "timeout";
        case CLASS_FAILURE:
            return "failure";
        default:
            return "unset";
    }
}

static int create_cache_files(const char *directory, int count,
                              char paths[MAX_WORKERS][4096]) {
    int index;

    for (index = 0; index < count; index++) {
        int fd;

        snprintf(paths[index], sizeof(paths[index]), "%s/cache-%d", directory,
                 index);
        fd = open(paths[index], O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) {
            return -1;
        }
        close(fd);
    }
    return 0;
}

static void remove_cache_files(char paths[MAX_WORKERS][4096], int count) {
    int index;

    for (index = 0; index < count; index++) {
        unlink(paths[index]);
    }
}

static int run_case(const char *name, int workers, int devices,
                    int unique_caches, int use_global_lock,
                    int expect_peer_pid) {
    char directory[] = "/tmp/hami-hostpid-race.XXXXXX";
    char cache_paths[MAX_WORKERS][4096] = {{0}};
    struct shared_state *state;
    pid_t children[MAX_WORKERS] = {0};
    int cache_count = unique_caches ? workers : 1;
    int own_count = 0;
    int peer_count = 0;
    int failure_count = 0;
    int timeout_count = 0;
    int index;
    int result = 0;

    if (workers <= 0 || workers > MAX_WORKERS || devices <= 0 ||
        devices > workers || mkdtemp(directory) == NULL) {
        return -1;
    }
    if (create_cache_files(directory, cache_count, cache_paths) != 0) {
        rmdir(directory);
        return -1;
    }
    state = mmap(NULL, sizeof(*state), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (state == MAP_FAILED) {
        remove_cache_files(cache_paths, cache_count);
        rmdir(directory);
        return -1;
    }
    memset(state, 0, sizeof(*state));

    for (index = 0; index < workers; index++) {
        const char *cache_path = cache_paths[unique_caches ? index : 0];

        children[index] = fork();
        if (children[index] == 0) {
            int status;

            hostpid_fallback_lock_after_fork();
            status = worker_main(state, directory, cache_path, index, workers,
                                 devices, unique_caches ? index : 0,
                                 use_global_lock,
                                 unique_caches && !use_global_lock);
            _exit(status);
        }
        if (children[index] < 0) {
            result = -1;
            break;
        }
    }
    atomic_store_explicit(&state->start, 1, memory_order_release);

    for (index = 0; index < workers; index++) {
        int status;

        if (children[index] <= 0 ||
            waitpid(children[index], &status, 0) != children[index]) {
            result = -1;
            continue;
        }
        switch (state->results[index].classification) {
            case CLASS_OWN_PID:
                own_count++;
                break;
            case CLASS_PEER_PID:
                peer_count++;
                break;
            case CLASS_TIMEOUT:
                timeout_count++;
                break;
            default:
                failure_count++;
                break;
        }
        printf("%s,%d,%d,%d,%" PRIdMAX ",%" PRIdMAX ",%s,%" PRId64
               ",%" PRId64 "\n",
               name, index,
               state->results[index].cache_index,
               state->results[index].device_index,
               (intmax_t)state->results[index].expected_pid,
               (intmax_t)state->results[index].observed_pid,
               classification_name(state->results[index].classification),
               state->results[index].lock_wait_us,
               state->results[index].discovery_us);
    }
    fprintf(stderr,
            "case=%s workers=%d own=%d peer=%d failure=%d timeout=%d\n",
            name, workers, own_count, peer_count, failure_count,
            timeout_count);

    if (expect_peer_pid) {
        if (peer_count == 0) {
            result = -1;
        }
    } else if (own_count != workers || peer_count != 0 ||
               failure_count != 0 || timeout_count != 0) {
        result = -1;
    }

    munmap(state, sizeof(*state));
    remove_cache_files(cache_paths, cache_count);
    rmdir(directory);
    return result;
}

int main(void) {
    int failures = 0;

    alarm(30);
    puts("case,worker,cache,device,expected_pid,observed_pid,classification,lock_wait_us,discovery_us");

    failures += run_case("legacy-one-worker", 1, 1, 0, 0, 0) != 0;
    failures += run_case("legacy-one-cache", 8, 1, 0, 0, 0) != 0;
    failures += run_case("legacy-independent-caches", 8, 1, 1, 0, 1) != 0;
    failures += run_case("legacy-balanced-devices", 8, 2, 1, 0, 1) != 0;
    failures += run_case("global-independent-caches", 8, 1, 1, 1, 0) != 0;
    failures += run_case("global-balanced-devices", 8, 2, 1, 1, 0) != 0;

    if (failures != 0) {
        fprintf(stderr, "%d host PID fallback race case(s) failed\n",
                failures);
        return 1;
    }
    fputs("host PID fallback race tests passed\n", stderr);
    return 0;
}

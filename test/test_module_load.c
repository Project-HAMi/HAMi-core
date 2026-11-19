/*
 * Module globals are allocated by the driver when a module is loaded, not
 * through cuMemAlloc. Run under LD_PRELOAD=libvgpu.so with a 2 GiB limit.
 *
 * usage: test_module_load [basic|limit|thread|process]
 *   basic:   free memory drops by the global on load, comes back on unload
 *   limit:   a load that does not fit in the limit fails with out of memory
 *   thread:  allocations of another thread are not charged to the module
 *   process: allocations of another process are not charged to the module
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

#define GLOBAL_MIB 256
#define CHURN_MIB 512
#define SLACK_MIB 16
#define ROUNDS 10

static const char ptx[] =
    ".version 6.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    ".visible .global .align 8 .b8 big[268435456];\n"
    ".visible .entry touch() {\n"
    "    .reg .u64 %rd<2>;\n"
    "    mov.u64 %rd1, big;\n"
    "    st.global.u8 [%rd1], 0;\n"
    "    ret;\n"
    "}\n";

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, (int)_e);                                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static atomic_int stop_churn;

static size_t free_mib(void) {
    size_t free = 0, total = 0;
    cuMemGetInfo(&free, &total);
    return free >> 20;
}

static void sleep_ms(int ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000};
    nanosleep(&ts, NULL);
}

static void *churn_thread(void *arg) {
    CUdeviceptr p;

    cuCtxSetCurrent((CUcontext)arg);
    while (!atomic_load_explicit(&stop_churn, memory_order_relaxed)) {
        if (cuMemAlloc(&p, (size_t)CHURN_MIB << 20) == CUDA_SUCCESS) {
            cuMemFree(p);
        }
    }
    return NULL;
}

/* A process without libvgpu that keeps allocating and freeing. It writes to
 * ready_fd after its first allocation succeeds. */
static int neighbor(int ready_fd) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr p;
    int ready = 0;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, TEST_DEVICE_ID));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    for (;;) {
        CHECK(cuMemAlloc(&p, (size_t)CHURN_MIB << 20));
        CHECK(cuMemFree(p));
        if (!ready) {
            ready = write(ready_fd, "r", 1) == 1;
            close(ready_fd);
        }
    }
}

/** Start the neighbor and wait until it is allocating. */
static pid_t start_neighbor(void) {
    int fds[2];
    char fd_arg[16], byte;
    struct pollfd pfd;
    pid_t pid;

    if (pipe(fds) != 0) {
        return -1;
    }
    pid = fork();
    if (pid == 0) {
        close(fds[0]);
        snprintf(fd_arg, sizeof(fd_arg), "%d", fds[1]);
        unsetenv("LD_PRELOAD");
        execl("/proc/self/exe", "test_module_load", "neighbor", fd_arg, (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    pfd.fd = fds[0];
    pfd.events = POLLIN;
    if (pid < 0 || poll(&pfd, 1, 30000) != 1 || read(fds[0], &byte, 1) != 1) {
        fprintf(stderr, "FAIL: neighbor process did not start allocating\n");
        if (pid > 0) {
            kill(pid, SIGKILL);
            waitpid(pid, NULL, 0);
        }
        pid = -1;
    }
    close(fds[0]);
    return pid;
}

/** Load and unload the module, checking what free memory reports. */
static int load_round(int round) {
    CUmodule module;
    size_t before = free_mib(), loaded, unloaded;

    CHECK(cuModuleLoadData(&module, ptx));
    loaded = free_mib();
    CHECK(cuModuleUnload(module));
    unloaded = free_mib();
    printf("round %d free MiB: before=%zu loaded=%zu unloaded=%zu\n",
           round, before, loaded, unloaded);
    if (before < loaded + GLOBAL_MIB || before > loaded + GLOBAL_MIB + SLACK_MIB) {
        fprintf(stderr, "FAIL: module charged %" PRId64 " MiB, expected %d\n",
                (int64_t)before - (int64_t)loaded, GLOBAL_MIB);
        return 1;
    }
    if (unloaded + SLACK_MIB < before) {
        fprintf(stderr, "FAIL: %zu MiB still charged after unload\n",
                before - unloaded);
        return 1;
    }
    return 0;
}

/** Module loads with another thread allocating; read free after it stops. */
static int thread_round(CUcontext ctx, int round) {
    pthread_t thread;
    CUmodule module;
    size_t before = free_mib(), loaded;

    atomic_store_explicit(&stop_churn, 0, memory_order_relaxed);
    if (pthread_create(&thread, NULL, churn_thread, ctx) != 0) {
        fprintf(stderr, "FAIL: pthread_create\n");
        return 1;
    }
    sleep_ms(20);
    CHECK(cuModuleLoadData(&module, ptx));
    atomic_store_explicit(&stop_churn, 1, memory_order_relaxed);
    pthread_join(thread, NULL);
    loaded = free_mib();
    CHECK(cuModuleUnload(module));
    printf("round %d free MiB: before=%zu loaded=%zu after=%zu\n",
           round, before, loaded, free_mib());
    if (before < loaded + GLOBAL_MIB || before > loaded + GLOBAL_MIB + SLACK_MIB) {
        fprintf(stderr, "FAIL: module charged %" PRId64 " MiB, expected %d\n",
                (int64_t)before - (int64_t)loaded, GLOBAL_MIB);
        return 1;
    }
    return 0;
}

static int limit_case(void) {
    CUmodule module = NULL;
    CUdeviceptr filler;
    size_t before = free_mib(), fill = before - GLOBAL_MIB / 2;
    CUresult res;

    CHECK(cuMemAlloc(&filler, fill << 20));
    res = cuModuleLoadData(&module, ptx);
    printf("load with %zu MiB free -> %d, free after %zu MiB\n",
           before - fill, (int)res, free_mib());
    if (res != CUDA_ERROR_OUT_OF_MEMORY || module != NULL) {
        fprintf(stderr, "FAIL: load over the limit returned %d\n", (int)res);
        return 1;
    }
    if (free_mib() + SLACK_MIB < before - fill) {
        fprintf(stderr, "FAIL: refused load is still charged\n");
        return 1;
    }
    CHECK(cuMemFree(filler));
    return load_round(0);
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "basic";
    CUdevice dev;
    CUcontext ctx;
    pid_t child = 0;
    int ret = 0;

    if (!strcmp(mode, "neighbor")) {
        return argc > 2 ? neighbor(atoi(argv[2])) : 1;
    }
    if (dlsym(RTLD_DEFAULT, "vgpu_hostpid_found") == NULL) {
        fprintf(stderr, "SKIP: libvgpu.so not preloaded\n");
        return 77;
    }
    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "SKIP: no usable GPU\n");
        return 77;
    }
    if (!strcmp(mode, "process")) {
        child = start_neighbor();
        if (child < 0) {
            return 1;
        }
    }
    if (cuDeviceGet(&dev, TEST_DEVICE_ID) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
        cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: context setup\n");
        ret = 1;
        goto out;
    }
    for (int round = 0; round < ROUNDS && ret == 0; round++) {
        if (!strcmp(mode, "limit")) {
            ret = limit_case();
            break;
        } else if (!strcmp(mode, "thread")) {
            ret = thread_round(ctx, round);
        } else {
            ret = load_round(round);
        }
    }
    cuDevicePrimaryCtxRelease(dev);
out:
    if (child > 0) {
        if (waitpid(child, NULL, WNOHANG) == child) {
            fprintf(stderr, "FAIL: neighbor process exited during the test\n");
            ret = 1;
        } else {
            kill(child, SIGKILL);
            waitpid(child, NULL, 0);
        }
    }
    if (ret == 0) {
        printf("test_module_load %s passed\n", mode);
    }
    return ret;
}

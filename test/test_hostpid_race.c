/*
 * Regression test: host PID discovery (set_task_pid, src/utils.c)
 *
 * Modes:
 *   neighbour <seconds> <mib>  Run without libvgpu. Repeatedly create the
 *                              primary context, allocate <mib>, hold 30 ms,
 *                              free and release, so a process keeps appearing
 *                              on and leaving the device.
 *   probe                      Run with libvgpu preloaded. cuInit runs host
 *                              PID discovery; then allocate 64 MiB and print
 *                              our PID next to the slice the hook reports.
 *
 * run_hostpid_race.sh compares each probe's PID with the hostPid libvgpu
 * logged (LIBCUDA_LOG_LEVEL=3). That needs getpid() to be the host PID, so
 * the wrapper skips inside a PID namespace.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

#define CHECK_DRV(call)                                                        \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            const char *_n = NULL;                                             \
            cuGetErrorName(_e, &_n);                                           \
            fprintf(stderr, "FATAL %s:%d: %s -> %d (%s)\n", __FILE__,          \
                    __LINE__, #call, (int)_e, _n ? _n : "?");                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int run_neighbour(double seconds, size_t mib) {
    struct timespec hold = {0, 30 * 1000 * 1000};
    double end = now_seconds() + seconds;
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr ptr;
    int cycles = 0;

    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuDeviceGet(&dev, TEST_DEVICE_ID));
    while (now_seconds() < end) {
        CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
        CHECK_DRV(cuCtxSetCurrent(ctx));
        CHECK_DRV(cuMemAlloc(&ptr, mib << 20));
        CHECK_DRV(cuMemsetD8(ptr, 1, mib << 20));
        CHECK_DRV(cuCtxSynchronize());
        nanosleep(&hold, NULL);
        CHECK_DRV(cuMemFree(ptr));
        CHECK_DRV(cuCtxSetCurrent(NULL));
        CHECK_DRV(cuDevicePrimaryCtxRelease(dev));
        cycles++;
    }
    printf("neighbour cycles=%d\n", cycles);
    return 0;
}

static int run_probe(void) {
    size_t free_b = 0, total_b = 0;
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr ptr;
    CUresult alloc;

    CHECK_DRV(cuInit(0));  // libvgpu discovers host PID
    CHECK_DRV(cuDeviceGet(&dev, TEST_DEVICE_ID));
    CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_DRV(cuCtxSetCurrent(ctx));
    CHECK_DRV(cuMemGetInfo(&free_b, &total_b));
    alloc = cuMemAlloc(&ptr, (size_t)64 << 20);
    printf("probe pid=%d slice_total_mib=%zu slice_used_mib=%zu alloc_64mib=%d\n",
           (int)getpid(), total_b >> 20, (total_b - free_b) >> 20, (int)alloc);
    if (alloc == CUDA_SUCCESS)
        cuMemFree(ptr);
    cuCtxSetCurrent(NULL);
    cuDevicePrimaryCtxRelease(dev);
    return alloc == CUDA_SUCCESS ? 0 : 3;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "probe") == 0)
        return run_probe();
    if (argc >= 4 && strcmp(argv[1], "neighbour") == 0)
        return run_neighbour(atof(argv[2]), (size_t)strtoul(argv[3], NULL, 10));
    fprintf(stderr, "usage: %s probe | neighbour <seconds> <mib>\n", argv[0]);
    return 2;
}

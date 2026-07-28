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
 * phase_probe - times each step of HAMi-core's host-PID detection.
 *
 * Replicates the sequence in src/utils.c:set_task_pid(), which is the work
 * held under the postinit lock, and reports the cost of each phase. This runs
 * standalone (no libvgpu.so interposed) so it measures the underlying driver
 * and NVML calls directly rather than the hooked versions.
 *
 * The point is to answer: of the serialized time per process, how much is
 * NVML enumeration versus CUDA context creation?
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <nvml.h>

#define MAX_PROC 1024

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void) {
    nvmlProcessInfo_t procs[MAX_PROC];
    nvmlDevice_t dev;
    unsigned int count;
    CUcontext ctx;
    double t[8];
    int i = 0;

    /* cuInit first: set_task_pid runs inside postInit, after the real
     * cuInit has already returned, so the driver is warm at this point. */
    t[i++] = now_ms();
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 1; }

    t[i++] = now_ms();
    if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 1; }

    t[i++] = now_ms();
    nvmlDeviceGetHandleByIndex(0, &dev);

    t[i++] = now_ms();
    count = MAX_PROC;
    nvmlDeviceGetComputeRunningProcesses(dev, &count, procs);   /* snapshot A */

    t[i++] = now_ms();
    cuDevicePrimaryCtxRetain(&ctx, 0);                          /* the probe */

    t[i++] = now_ms();
    count = MAX_PROC;
    nvmlDeviceGetComputeRunningProcesses(dev, &count, procs);   /* snapshot B */

    t[i++] = now_ms();
    cuDevicePrimaryCtxRelease(0);

    t[i++] = now_ms();

    printf("{\"cuInit_ms\":%.2f,"
           "\"nvmlInit_ms\":%.2f,"
           "\"nvmlGetHandle_ms\":%.2f,"
           "\"snapshotA_ms\":%.2f,"
           "\"ctxRetain_ms\":%.2f,"
           "\"snapshotB_ms\":%.2f,"
           "\"ctxRelease_ms\":%.2f,"
           "\"critical_section_ms\":%.2f,"
           "\"procs_seen\":%u}\n",
           t[1] - t[0], t[2] - t[1], t[3] - t[2], t[4] - t[3],
           t[5] - t[4], t[6] - t[5], t[7] - t[6],
           t[7] - t[1],   /* everything after cuInit == the locked region */
           count);
    return 0;
}

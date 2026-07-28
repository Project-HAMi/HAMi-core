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
 * ctx_experiment - is cuDevicePrimaryCtxRetain's cost inherent, or contention?
 *
 * Answers two questions raised on Project-HAMi/HAMi#1662:
 *
 *   1. Does another process holding the primary context open make this
 *      process's cuDevicePrimaryCtxRetain slower? Run this under the three
 *      holder conditions (none / primary-context holder / non-primary-context
 *      holder) and compare.
 *
 *   2. Can the probe cost be amortized? Primary contexts are refcounted per
 *      process, so a retain after a prior retain in the same process should
 *      be near-free. If so, the expensive part is first-context creation and
 *      the Release in set_task_pid() throws that work away, since the
 *      application will create its own context moments later.
 *
 * A warmup retain/release pair runs first to take GPU wake latency out of the
 * measurement, then three retain/release cycles are timed.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *label = argc > 1 ? argv[1] : "unlabelled";
    CUcontext ctx;
    double a, b;
    double ret[3], rel[3];

    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed\n");
        return 1;
    }

    /* Warmup: wakes the GPU if it was runtime-suspended, so the numbers below
     * reflect steady state rather than a one-off power transition. */
    a = now_ms();
    cuDevicePrimaryCtxRetain(&ctx, 0);
    cuDevicePrimaryCtxRelease(0);
    double warmup_ms = now_ms() - a;

    for (int i = 0; i < 3; i++) {
        a = now_ms();
        cuDevicePrimaryCtxRetain(&ctx, 0);
        b = now_ms();
        cuDevicePrimaryCtxRelease(0);
        ret[i] = b - a;
        rel[i] = now_ms() - b;
    }

    printf("{\"cond\":\"%s\",\"warmup_pair_ms\":%.2f,"
           "\"retain1_ms\":%.2f,\"retain2_ms\":%.2f,\"retain3_ms\":%.2f,"
           "\"release1_ms\":%.2f,\"release2_ms\":%.2f,\"release3_ms\":%.2f}\n",
           label, warmup_ms,
           ret[0], ret[1], ret[2], rel[0], rel[1], rel[2]);
    return 0;
}

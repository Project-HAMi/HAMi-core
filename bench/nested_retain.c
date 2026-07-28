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
 * nested_retain - is the primary context refcount cheap to bump once held?
 *
 * set_task_pid() does retain -> probe -> release. The application then creates
 * its own context moments later and pays full construction cost again.
 *
 * If a nested retain (one taken while the context is still held) is near-free,
 * then the Release is discarding work the process is about to redo, and the
 * probe could hand its context off to the application instead of tearing it
 * down. This measures that.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>
#include <cuda.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void) {
    CUcontext c1, c2, c3;
    double a;
    double first, nested, after_release;
    CUresult r;

    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed\n");
        return 1;
    }

    /* Warmup so GPU wake latency is not attributed to the first retain. */
    if ((r = cuDevicePrimaryCtxRetain(&c1, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "warmup retain failed: %d\n", r);
        return 1;
    }
    if ((r = cuDevicePrimaryCtxRelease(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "warmup release failed: %d\n", r);
        return 1;
    }

    /* Cold retain: nothing held by this process. */
    a = now_ms();
    if ((r = cuDevicePrimaryCtxRetain(&c1, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "cold retain failed: %d\n", r);
        return 1;
    }
    first = now_ms() - a;

    /* Nested retain: context still held, so this should be refcount only. */
    a = now_ms();
    if ((r = cuDevicePrimaryCtxRetain(&c2, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "nested retain failed: %d\n", r);
        return 1;
    }
    nested = now_ms() - a;

    if ((r = cuDevicePrimaryCtxRelease(0)) != CUDA_SUCCESS) {   /* drop to refcount 1, context stays alive */
        fprintf(stderr, "release failed: %d\n", r);
        return 1;
    }

    /* Retain again while still held. Models the application acquiring a
     * context after the probe finished but did not fully release. */
    a = now_ms();
    if ((r = cuDevicePrimaryCtxRetain(&c3, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "retain-while-held failed: %d\n", r);
        return 1;
    }
    after_release = now_ms() - a;

    if ((r = cuDevicePrimaryCtxRelease(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "final release (1/2) failed: %d\n", r);
        return 1;
    }
    if ((r = cuDevicePrimaryCtxRelease(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "final release (2/2) failed: %d\n", r);
        return 1;
    }

    printf("{\"cold_retain_ms\":%.3f,\"nested_retain_ms\":%.3f,"
           "\"retain_while_held_ms\":%.3f,\"same_ctx\":%s}\n",
           first, nested, after_release,
           (c1 == c2 && c2 == c3) ? "true" : "false");
    return 0;
}

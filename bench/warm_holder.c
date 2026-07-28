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
 * warm_holder - holds a CUDA primary context open so the GPU does not
 * runtime-suspend between benchmark iterations.
 *
 * On an Optimus laptop the discrete GPU powers down when idle, and the next
 * cuInit() pays roughly 1.5s of wake latency. That artifact dwarfs the effect
 * under study. Server GPUs in the scenario this benchmark models are always
 * resident, so holding a context is the faithful configuration, not a cheat.
 *
 * Runs until killed. Deliberately does no allocation beyond the context so it
 * costs a fixed, small amount of device memory.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cuda.h>

/*
 * Modes:
 *   primary    (default) retain the device primary context, the same object
 *              set_task_pid() probes with
 *   nonprimary create an ordinary context instead, which keeps the GPU awake
 *              without holding the primary context. Used to separate "GPU is
 *              powered up" from "another process holds the primary context".
 */
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "primary";
    CUcontext ctx;
    CUdevice dev;
    CUresult r;

    if ((r = cuInit(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "warm_holder: cuInit failed: %d\n", r);
        return 1;
    }

    if (strcmp(mode, "primary") == 0) {
        if ((r = cuDevicePrimaryCtxRetain(&ctx, 0)) != CUDA_SUCCESS) {
            fprintf(stderr, "warm_holder: ctx retain failed: %d\n", r);
            return 1;
        }
    } else if (strcmp(mode, "nonprimary") == 0) {
        if ((r = cuDeviceGet(&dev, 0)) != CUDA_SUCCESS ||
            (r = cuCtxCreate_v2(&ctx, 0, dev)) != CUDA_SUCCESS) {
            fprintf(stderr, "warm_holder: ctx create failed: %d\n", r);
            return 1;
        }
    } else {
        fprintf(stderr, "warm_holder: unknown mode '%s' (expected 'primary' or 'nonprimary')\n", mode);
        return 2;
    }

    printf("warm_holder ready mode=%s pid=%d\n", mode, getpid());
    fflush(stdout);

    for (;;) pause();
    return 0;
}

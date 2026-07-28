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

/* Verifies _Atomic uint64_t did not change shared_region_t's layout or size,
 * which matters since this struct is persisted to /tmp/cudevshr.cache and
 * read from a separate Go process via a hand-written mirror struct
 * (pkg/monitor/nvidia/v1/spec.go:sharedRegionT in the HAMi repo).
 *
 * The expected values below were captured by compiling that Go struct with
 * unsafe.Sizeof/unsafe.Offsetof on linux/amd64. If this check fails, either
 * the C struct changed and the Go mirror needs updating, or vice versa. */
#include <stdio.h>
#include <stddef.h>
#include "../src/multiprocess/multiprocess_memory_limit.h"

#define EXPECT_SIZEOF_SHARED_REGION 2008952
#define EXPECT_OFFSETOF_LIMIT       1600
#define EXPECT_OFFSETOF_SM_LIMIT    1728
#define EXPECT_OFFSETOF_PROCS       1856
#define EXPECT_SIZEOF_LIMIT_ELEM    8

static int check(const char *name, size_t got, size_t want) {
    int ok = (got == want);
    printf("%-24s = %-10zu (expected %zu) %s\n", name, got, want, ok ? "OK" : "MISMATCH");
    return ok;
}

int main(void) {
    int ok = 1;
    ok &= check("sizeof(shared_region_t)", sizeof(shared_region_t), EXPECT_SIZEOF_SHARED_REGION);
    ok &= check("offsetof(limit)", offsetof(shared_region_t, limit), EXPECT_OFFSETOF_LIMIT);
    ok &= check("offsetof(sm_limit)", offsetof(shared_region_t, sm_limit), EXPECT_OFFSETOF_SM_LIMIT);
    ok &= check("offsetof(procs)", offsetof(shared_region_t, procs), EXPECT_OFFSETOF_PROCS);
    ok &= check("sizeof(limit[0])", sizeof(((shared_region_t*)0)->limit[0]), EXPECT_SIZEOF_LIMIT_ELEM);

    if (!ok) {
        fprintf(stderr, "abi_check: shared_region_t layout diverged from the Go mirror struct\n");
        return 1;
    }
    return 0;
}

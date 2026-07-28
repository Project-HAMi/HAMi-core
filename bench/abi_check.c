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
 * read from a separate Go process via a hand-written mirror struct. */
#include <stdio.h>
#include <stddef.h>
#include "../src/multiprocess/multiprocess_memory_limit.h"

int main(void) {
    printf("sizeof(shared_region_t) = %zu\n", sizeof(shared_region_t));
    printf("offsetof(limit)         = %zu\n", offsetof(shared_region_t, limit));
    printf("offsetof(sm_limit)      = %zu\n", offsetof(shared_region_t, sm_limit));
    printf("offsetof(procs)         = %zu\n", offsetof(shared_region_t, procs));
    printf("sizeof(limit[0])        = %zu\n", sizeof(((shared_region_t*)0)->limit[0]));
    return 0;
}

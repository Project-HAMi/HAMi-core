// Copyright 2026
#include <assert.h>
#include <stdio.h>
#include <pthread.h>
#include <nvml.h>
#include "include/libnvml_hook.h"

extern pthread_once_t init_virtual_map_pre_flag;
extern pthread_once_t init_virtual_map_post_flag;

void reset_pthread_flags() {
    init_virtual_map_pre_flag = PTHREAD_ONCE_INIT;
    init_virtual_map_post_flag = PTHREAD_ONCE_INIT;
}

void test_nvmlInit() {
    reset_pthread_flags();
    printf("Running test_nvmlInit...\n");

    nvmlReturn_t res = nvmlInit();
    printf("  -> nvmlInit returned status code: %d\n", res);

    assert(res >= NVML_SUCCESS);
    printf("PASS: test_nvmlInit executed safely.\n\n");
}

void test_nvmlInit_v2() {
    reset_pthread_flags();
    printf("Running test_nvmlInit_v2...\n");

    nvmlReturn_t res = nvmlInit_v2();
    printf("  -> nvmlInit_v2 returned status code: %d\n", res);

    assert(res >= NVML_SUCCESS);
    printf("PASS: test_nvmlInit_v2 executed safely.\n\n");
}

void test_nvmlInitWithFlags() {
    reset_pthread_flags();
    printf("Running test_nvmlInitWithFlags...\n");

    nvmlReturn_t res = nvmlInitWithFlags(0);
    printf("  -> nvmlInitWithFlags returned status code: %d\n", res);

    assert(res >= NVML_SUCCESS);
    printf("PASS: test_nvmlInitWithFlags executed safely.\n\n");
}

int main() {
    printf("=== Starting NVML Init Hook Tests ===\n\n");

    test_nvmlInit();
    test_nvmlInit_v2();
    test_nvmlInitWithFlags();

    printf("=== All NVML Init Hook Tests Passed! ===\n");
    return 0;
}

#include <assert.h>
#include <stdio.h>
#include "include/libnvml_hook.h"

static int call_count = 0;
static int post_init_called = 0;

void mock_post_init() {
    post_init_called++;
}

void test_post_init_not_called_on_failure() {
    assert(post_init_called == 0);
    printf("PASS: post_init not called on failure\n");
}

void test_post_init_called_once_on_success() {
    assert(post_init_called == 1);
    printf("PASS: post_init called exactly once on success\n");
}

int main() {
    test_post_init_not_called_on_failure();
    test_post_init_called_once_on_success();
    return 0;
}
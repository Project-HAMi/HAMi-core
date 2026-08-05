/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>

#include "../src/cuda/context_accounting.h"

#define CONTEXT_BYTES 436207616UL

static void test_nested_lifetime(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;

    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(state.retain_count == 1);

    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == 0);
    assert(state.retain_count == 2);

    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == 0);
    assert(state.retain_count == 1);

    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == 0);
    assert(state.retain_count == 2);

    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == 0);
    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(state.retain_count == 0);
}

static void test_size_can_be_charged_late(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 99;

    assert(primary_context_record_retain(&state, 0, &bytes) == 0);
    assert(bytes == 0);
    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == 0);
    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
}

static void test_failed_add_is_not_removed(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;

    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    primary_context_cancel_charge(&state);
    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == 0);
}

static void test_failed_remove_is_retried(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;

    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);

    primary_context_restore_charge(&state, bytes);
    assert(state.charged_bytes == CONTEXT_BYTES);
    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == 0);
    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
}

static void test_rejects_invalid_calls(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;

    errno = 0;
    assert(primary_context_record_release(&state, &bytes) == -1);
    assert(errno == EINVAL);

    errno = 0;
    assert(primary_context_record_retain(NULL, 1, &bytes) == -1);
    assert(errno == EINVAL);

    state.retain_count = UINT_MAX;
    errno = 0;
    assert(primary_context_record_retain(&state, 1, &bytes) == -1);
    assert(errno == EOVERFLOW);
}

int main(void) {
    test_nested_lifetime();
    test_size_can_be_charged_late();
    test_failed_add_is_not_removed();
    test_failed_remove_is_retried();
    test_rejects_invalid_calls();
    puts("context accounting tests passed");
    return 0;
}

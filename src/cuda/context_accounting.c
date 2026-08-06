/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#include "cuda/context_accounting.h"

#include <errno.h>
#include <limits.h>

static int record_retain(primary_context_accounting_t *state,
                         size_t context_bytes, size_t *bytes_to_add,
                         int require_charge) {
    if (state == NULL || bytes_to_add == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (state->retain_count == UINT_MAX) {
        errno = EOVERFLOW;
        return -1;
    }
    if (require_charge && state->charged_bytes == 0 && context_bytes == 0) {
        errno = ENODATA;
        return -1;
    }

    *bytes_to_add = 0;
    if (state->charged_bytes == 0 && context_bytes > 0) {
        state->charged_bytes = context_bytes;
        *bytes_to_add = context_bytes;
    }
    state->retain_count++;
    return 0;
}

int primary_context_record_retain(primary_context_accounting_t *state,
                                  size_t context_bytes,
                                  size_t *bytes_to_add) {
    return record_retain(state, context_bytes, bytes_to_add, 0);
}

int primary_context_record_accounted_retain(
    primary_context_accounting_t *state, size_t context_bytes,
    size_t *bytes_to_add) {
    return record_retain(state, context_bytes, bytes_to_add, 1);
}

int primary_context_record_release(primary_context_accounting_t *state,
                                   size_t *bytes_to_remove) {
    if (state == NULL || bytes_to_remove == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (state->retain_count == 0) {
        errno = EINVAL;
        return -1;
    }

    *bytes_to_remove = 0;
    state->retain_count--;
    if (state->retain_count == 0) {
        *bytes_to_remove = state->charged_bytes;
        state->charged_bytes = 0;
    }
    return 0;
}

int primary_context_rollback_retain(primary_context_accounting_t *state,
                                    size_t bytes_to_add) {
    if (state == NULL || state->retain_count == 0) {
        errno = EINVAL;
        return -1;
    }
    if (bytes_to_add > 0 && state->charged_bytes != bytes_to_add) {
        errno = EINVAL;
        return -1;
    }

    state->retain_count--;
    if (bytes_to_add > 0) {
        state->charged_bytes = 0;
    }
    return 0;
}

void primary_context_restore_charge(primary_context_accounting_t *state,
                                    size_t context_bytes) {
    if (state != NULL && state->retain_count == 0 && context_bytes > 0) {
        state->charged_bytes = context_bytes;
    }
}

void primary_context_accounting_reset(primary_context_accounting_t *states,
                                      size_t state_count) {
    size_t i;

    if (states == NULL) {
        return;
    }
    for (i = 0; i < state_count; i++) {
        states[i].retain_count = 0;
        states[i].charged_bytes = 0;
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#ifndef SRC_CUDA_CONTEXT_ACCOUNTING_H_
#define SRC_CUDA_CONTEXT_ACCOUNTING_H_

#include <stddef.h>

typedef struct {
    unsigned int retain_count;
    size_t charged_bytes;
} primary_context_accounting_t;

/*
 * Record a successful retain or release. A known size is charged once and
 * removed when the final retain is released.
 */
int primary_context_record_retain(primary_context_accounting_t *state,
                                  size_t context_bytes,
                                  size_t *bytes_to_add);

/*
 * Record a retain only when its context memory is already charged or a
 * nonzero charge can be applied before the caller reports success.
 */
int primary_context_record_accounted_retain(
    primary_context_accounting_t *state, size_t context_bytes,
    size_t *bytes_to_add);
int primary_context_record_release(primary_context_accounting_t *state,
                                   size_t *bytes_to_remove);

/* Undo the most recent retain after its external accounting step fails. */
int primary_context_rollback_retain(primary_context_accounting_t *state,
                                    size_t bytes_to_add);

/* Restore a charge that could not be removed from shared accounting. */
void primary_context_restore_charge(primary_context_accounting_t *state,
                                    size_t context_bytes);

/* Clear process-local accounting inherited across fork(). */
void primary_context_accounting_reset(primary_context_accounting_t *states,
                                      size_t state_count);

#endif  // SRC_CUDA_CONTEXT_ACCOUNTING_H_

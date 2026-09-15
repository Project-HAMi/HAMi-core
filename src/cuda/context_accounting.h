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

int primary_context_record_release(primary_context_accounting_t *state,
                                   size_t *bytes_to_remove);

/* Registers the fork handlers that reset accounting in the child. */
int context_accounting_register_fork_handlers(void);

#endif  // SRC_CUDA_CONTEXT_ACCOUNTING_H_

#ifndef SRC_CUDA_CONTEXT_FORK_H_
#define SRC_CUDA_CONTEXT_FORK_H_

int context_accounting_register_fork_handlers(void);
void context_accounting_fork_prepare(void);
void context_accounting_fork_parent(void);
void context_accounting_fork_child(void);

#endif  // SRC_CUDA_CONTEXT_FORK_H_

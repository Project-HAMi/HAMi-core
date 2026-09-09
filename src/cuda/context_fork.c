#include "cuda/context_fork.h"

#include <pthread.h>

extern pthread_once_t post_cuinit_flag;
extern int pidfound;

static void childReinitPostInit(void) {
    context_accounting_fork_child();
    post_cuinit_flag = (pthread_once_t)PTHREAD_ONCE_INIT;
    pidfound = 0;
}

int context_accounting_register_fork_handlers(void) {
    return pthread_atfork(context_accounting_fork_prepare,
                          context_accounting_fork_parent,
                          childReinitPostInit);
}

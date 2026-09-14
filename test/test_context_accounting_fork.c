/* Exercise the same registration and child callback used by preInit(). */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <signal.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#include "context_hook_fixture.h"

int context_accounting_register_fork_handlers(void);

static int initialized_count;

static void mark_initialized(void) {
    initialized_count++;
}

static void fork_completes(void) {
    int status;
    pid_t child = fork();

    assert(child >= 0);
    if (child == 0) {
        _exit(0);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

int main(void) {
    CUcontext ctx;
    int status;
    pid_t child;

    alarm(10);
    assert(context_accounting_register_fork_handlers() == 0);
    assert(pthread_once(&post_cuinit_flag, mark_initialized) == 0);
    assert(initialized_count == 1);
    assert(cuDevicePrimaryCtxRetain(&ctx, 0) == CUDA_SUCCESS);
    assert(cuDevicePrimaryCtxRetain(&ctx, 0) == CUDA_SUCCESS);
    assert(cuDevicePrimaryCtxRetain(&ctx, 3) == CUDA_SUCCESS);
    assert(context_charge[0] == TEST_CONTEXT_BYTES);
    assert(context_charge[3] == TEST_CONTEXT_BYTES);

    child = fork();
    assert(child >= 0);
    if (child == 0) {
        alarm(5);
        assert(pidfound == 0);
        assert(context_size == TEST_CONTEXT_BYTES);
        assert(pthread_once(&post_cuinit_flag, mark_initialized) == 0);
        assert(initialized_count == 2);

        /* The child's external driver state and shared-region slot are new.
         * Leave production accounting untouched: the registered callback must
         * have cleared it before either of these retains. */
        memset(driver_refs, 0, sizeof(driver_refs));
        memset(context_charge, 0, sizeof(context_charge));
        assert(cuDevicePrimaryCtxRetain(&ctx, 0) == CUDA_SUCCESS);
        assert(cuDevicePrimaryCtxRetain(&ctx, 3) == CUDA_SUCCESS);
        assert(context_charge[0] == TEST_CONTEXT_BYTES);
        assert(context_charge[3] == TEST_CONTEXT_BYTES);
        assert(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
        assert(cuDevicePrimaryCtxRelease_v2(3) == CUDA_SUCCESS);
        assert(context_charge[0] == 0 && context_charge[3] == 0);
        fork_completes();
        _exit(0);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(pidfound == 1);
    assert(pthread_once(&post_cuinit_flag, mark_initialized) == 0);
    assert(initialized_count == 1);
    assert(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
    assert(context_charge[0] == TEST_CONTEXT_BYTES);
    assert(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
    assert(cuDevicePrimaryCtxRelease_v2(3) == CUDA_SUCCESS);
    assert(context_charge[0] == 0 && context_charge[3] == 0);
    fork_completes();
    alarm(0);
    puts("context accounting production fork registration tests passed");
    return 0;
}

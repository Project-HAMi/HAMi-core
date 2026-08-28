/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

/*
 * GPU-free test of the fork handlers in src/cuda/context.c, driven through a
 * real pthread_atfork registration and a real fork(), the way libvgpu.c wires
 * them.  test_context_accounting covers what primary_context_accounting_reset
 * does to the state; this covers that the child handler actually runs after
 * fork, that it leaves the accounting mutex usable in the child, and that the
 * parent handler releases it again in the parent.
 *
 * The private arrays in context.c are static, so the handlers are observed
 * through what they touch that is visible: ctx_activate, which the child
 * handler clears for every device, and context_size, which it must keep.  A
 * mutex left held would deadlock the next fork's prepare handler, so each
 * process forks once more under an alarm that turns a hang into a failure.
 */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

/* The handlers under test.  libvgpu.c registers the same prepare and parent
 * handlers, and its child handler calls context_accounting_fork_child first. */
void context_accounting_fork_prepare(void);
void context_accounting_fork_parent(void);
void context_accounting_fork_child(void);

/* Defined in multiprocess_memory_limit.c in production.  The handlers read
 * and write these, so the test owns them here. */
int ctx_activate[CUDA_DEVICE_MAX_COUNT];
size_t context_size;

#define PROBED_SIZE 436207616UL
#define FORK_TIMEOUT_S 5

typedef struct {
    int activate[CUDA_DEVICE_MAX_COUNT];
    size_t size;
    int refork_ok;
} child_view_t;

static void on_alarm(int sig) {
    (void)sig;
    _exit(3);
}

/* A held accounting mutex deadlocks the prepare handler of the next fork.
 * The alarm turns that into exit 3 instead of a hang. */
static int fork_completes(void) {
    pid_t pid;
    int status;

    alarm(FORK_TIMEOUT_S);
    pid = fork();
    if (pid == 0) {
        _exit(0);
    }
    alarm(0);
    if (pid < 0) {
        return 0;
    }
    return waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
           WEXITSTATUS(status) == 0;
}

int main(void) {
    child_view_t seen;
    int result_pipe[2];
    pid_t child;
    int status;
    int dev;

    signal(SIGALRM, on_alarm);
    assert(pthread_atfork(context_accounting_fork_prepare,
                          context_accounting_fork_parent,
                          context_accounting_fork_child) == 0);

    for (dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        ctx_activate[dev] = dev + 1;
    }
    context_size = PROBED_SIZE;
    assert(pipe(result_pipe) == 0);

    alarm(FORK_TIMEOUT_S);
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        alarm(0);
        close(result_pipe[0]);
        memcpy(seen.activate, ctx_activate, sizeof(seen.activate));
        seen.size = context_size;
        seen.refork_ok = fork_completes();
        if (write(result_pipe[1], &seen, sizeof(seen)) !=
            (ssize_t)sizeof(seen)) {
            _exit(2);
        }
        _exit(0);
    }
    alarm(0);

    close(result_pipe[1]);
    assert(read(result_pipe[0], &seen, sizeof(seen)) == (ssize_t)sizeof(seen));
    close(result_pipe[0]);
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 0);

    /* The child handler ran: every device cleared, the probed size kept, and
     * the child could fork again, so its accounting mutex was released. */
    for (dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        assert(seen.activate[dev] == 0);
    }
    assert(seen.size == PROBED_SIZE);
    assert(seen.refork_ok == 1);

    /* The parent is untouched, and its handler released the mutex too. */
    for (dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        assert(ctx_activate[dev] == dev + 1);
    }
    assert(context_size == PROBED_SIZE);
    assert(fork_completes() == 1);

    puts("context accounting fork wiring test passed");
    return 0;
}

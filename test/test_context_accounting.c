#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

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

static void test_required_charge_fails_without_size(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 99;

    errno = 0;
    assert(primary_context_record_accounted_retain(&state, 0, &bytes) == -1);
    assert(errno == ENODATA);
    assert(state.retain_count == 0);
    assert(state.charged_bytes == 0);
    assert(bytes == 99);
}

static void test_unmeasured_context_defers_instead_of_failing(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 99;

    /* Mirrors the retain path when the context size cannot be measured. */
    errno = 0;
    assert(primary_context_record_accounted_retain(&state, 0, &bytes) == -1);
    assert(errno == ENODATA);
    assert(primary_context_record_retain(&state, 0, &bytes) == 0);
    assert(bytes == 0);
    assert(state.retain_count == 1);
    assert(state.charged_bytes == 0);

    /* A later retain that knows the size still charges exactly once. */
    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(state.retain_count == 2);
    assert(state.charged_bytes == CONTEXT_BYTES);
}

static void test_accounted_nested_retain_reuses_existing_charge(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;

    assert(primary_context_record_accounted_retain(
               &state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(primary_context_record_accounted_retain(&state, 0, &bytes) == 0);
    assert(bytes == 0);
    assert(state.retain_count == 2);
    assert(state.charged_bytes == CONTEXT_BYTES);

    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == 0);
    assert(primary_context_record_release(&state, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
}

static void test_failed_add_rolls_back_retain(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;

    assert(primary_context_record_accounted_retain(
               &state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(primary_context_rollback_retain(&state, bytes) == 0);
    assert(state.retain_count == 0);
    assert(state.charged_bytes == 0);
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

    state.retain_count = 0;
    errno = 0;
    assert(primary_context_rollback_retain(&state, 0) == -1);
    assert(errno == EINVAL);
}

static void test_failed_charge_keeps_the_retain(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;
    size_t retried = 99;

    /* Mirrors the retain path when the shared region refuses the charge. */
    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(primary_context_rollback_retain(&state, bytes) == 0);
    assert(primary_context_record_retain(&state, 0, &retried) == 0);
    assert(retried == 0);
    assert(state.retain_count == 1);
    assert(state.charged_bytes == 0);

    /* A later retain can still charge it once. */
    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    assert(bytes == CONTEXT_BYTES);
    assert(state.charged_bytes == CONTEXT_BYTES);
}

static void test_rollback_rejects_mismatched_charge(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;

    assert(primary_context_record_retain(&state, CONTEXT_BYTES, &bytes) == 0);
    errno = 0;
    assert(primary_context_rollback_retain(&state, CONTEXT_BYTES + 1) == -1);
    assert(errno == EINVAL);
    assert(state.retain_count == 1);
    assert(state.charged_bytes == CONTEXT_BYTES);
}

static void test_restore_is_ignored_while_retained(void) {
    primary_context_accounting_t state = {0};
    size_t bytes = 0;

    /* restore_charge is only for a fully released context whose removal
     * failed; with a retain outstanding it must not touch the charge. */
    assert(primary_context_record_retain(&state, 0, &bytes) == 0);
    primary_context_restore_charge(&state, CONTEXT_BYTES);
    assert(state.charged_bytes == 0);
    assert(state.retain_count == 1);
}

static void test_nonzero_device_accounting_is_isolated(void) {
    primary_context_accounting_t states[4] = {{0}};
    size_t bytes = 0;
    const int dev = 3;

    assert(primary_context_record_accounted_retain(
               &states[dev], CONTEXT_BYTES, &bytes) == 0);
    assert(states[dev].retain_count == 1);
    assert(states[dev].charged_bytes == CONTEXT_BYTES);
    assert(states[0].retain_count == 0);
    assert(states[0].charged_bytes == 0);
}

static void test_forked_child_resets_private_accounting(void) {
    primary_context_accounting_t states[4] = {{0}};
    primary_context_accounting_t child_states[4] = {{0}};
    size_t bytes = 0;
    int result_pipe[2];
    pid_t child;
    int status;

    assert(primary_context_record_accounted_retain(
               &states[0], CONTEXT_BYTES, &bytes) == 0);
    assert(primary_context_record_accounted_retain(
               &states[3], CONTEXT_BYTES, &bytes) == 0);
    assert(pipe(result_pipe) == 0);

    child = fork();
    assert(child >= 0);
    if (child == 0) {
        close(result_pipe[0]);
        primary_context_accounting_reset(states, 4);
        if (write(result_pipe[1], states, sizeof(states)) !=
            (ssize_t)sizeof(states)) {
            _exit(2);
        }
        _exit(0);
    }

    close(result_pipe[1]);
    assert(read(result_pipe[0], child_states, sizeof(child_states)) ==
           (ssize_t)sizeof(child_states));
    close(result_pipe[0]);
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 0);

    assert(child_states[0].retain_count == 0);
    assert(child_states[0].charged_bytes == 0);
    assert(child_states[3].retain_count == 0);
    assert(child_states[3].charged_bytes == 0);
    assert(states[0].retain_count == 1);
    assert(states[0].charged_bytes == CONTEXT_BYTES);
    assert(states[3].retain_count == 1);
    assert(states[3].charged_bytes == CONTEXT_BYTES);
}

int main(void) {
    test_nested_lifetime();
    test_size_can_be_charged_late();
    test_unmeasured_context_defers_instead_of_failing();
    test_required_charge_fails_without_size();
    test_accounted_nested_retain_reuses_existing_charge();
    test_failed_add_rolls_back_retain();
    test_failed_remove_is_retried();
    test_rejects_invalid_calls();
    test_rollback_rejects_mismatched_charge();
    test_restore_is_ignored_while_retained();
    test_failed_charge_keeps_the_retain();
    test_nonzero_device_accounting_is_isolated();
    test_forked_child_resets_private_accounting();
    puts("context accounting tests passed");
    return 0;
}

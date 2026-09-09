#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../src/include/hostpid_fallback_lock.h"

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
static int paused;
static int resume_acquire;
static int hook_calls;
static int acquiring_fd;
static int acquire_result;
static const char *lock_path;

static void pause_first_acquire(void) {
    assert(pthread_mutex_lock(&mutex) == 0);
    if (++hook_calls == 1) {
        acquiring_fd = hostpid_fallback_lock_active_fd();
        paused = 1;
        assert(pthread_cond_broadcast(&condition) == 0);
        while (!resume_acquire) {
            assert(pthread_cond_wait(&condition, &mutex) == 0);
        }
    }
    assert(pthread_mutex_unlock(&mutex) == 0);
}

static void *acquire_in_thread(void *unused) {
    (void)unused;
    acquire_result = hostpid_fallback_lock_acquire_at(lock_path, getuid(), 5000);
    return NULL;
}

static pthread_t start_paused_acquire(void) {
    pthread_t thread;

    paused = 0;
    resume_acquire = 0;
    hook_calls = 0;
    acquire_result = -1;
    hostpid_fallback_lock_set_before_flock_hook(pause_first_acquire);
    assert(pthread_create(&thread, NULL, acquire_in_thread, NULL) == 0);
    assert(pthread_mutex_lock(&mutex) == 0);
    while (!paused) {
        assert(pthread_cond_wait(&condition, &mutex) == 0);
    }
    assert(pthread_mutex_unlock(&mutex) == 0);
    assert(acquiring_fd >= 0);
    return thread;
}

static void resume_thread(void) {
    assert(pthread_mutex_lock(&mutex) == 0);
    resume_acquire = 1;
    assert(pthread_cond_broadcast(&condition) == 0);
    assert(pthread_mutex_unlock(&mutex) == 0);
}

static void test_release_cannot_close_an_inflight_acquire(void) {
    pthread_t thread = start_paused_acquire();
    int contender;

    errno = 0;
    assert(hostpid_fallback_lock_release() == -1 && errno == EBUSY);
    assert(fcntl(acquiring_fd, F_GETFD) >= 0);
    errno = 0;
    assert(hostpid_fallback_lock_acquire_at(lock_path, getuid(), 100) == -1 &&
           errno == EBUSY);
    resume_thread();
    assert(pthread_join(thread, NULL) == 0);
    assert(acquire_result == 0);
    assert(hostpid_fallback_lock_active_fd() == acquiring_fd);
    contender = open(lock_path, O_RDONLY | O_DIRECTORY);
    assert(contender >= 0);
    assert(flock(contender, LOCK_EX | LOCK_NB) == -1);
    assert(errno == EWOULDBLOCK || errno == EAGAIN);
    /* A different thread may release a completed acquisition. */
    assert(hostpid_fallback_lock_release() == 0);
    assert(flock(contender, LOCK_EX | LOCK_NB) == 0);
    close(contender);
}

static void test_cancelled_acquire_does_not_strand_the_guard(void) {
    pthread_t thread = start_paused_acquire();
    void *result = NULL;

    assert(pthread_cancel(thread) == 0);
    resume_thread();
    assert(pthread_join(thread, &result) == 0);
    assert(result == PTHREAD_CANCELED);
    assert(hostpid_fallback_lock_active_fd() == -1);
    errno = 0;
    assert(fcntl(acquiring_fd, F_GETFD) == -1 && errno == EBADF);
    hostpid_fallback_lock_set_before_flock_hook(NULL);
    assert(hostpid_fallback_lock_acquire_at(lock_path, getuid(), 1000) == 0);
    assert(hostpid_fallback_lock_release() == 0);
}

static void test_fork_child_clears_an_inherited_busy_guard(void) {
    pthread_t thread = start_paused_acquire();
    pid_t child = fork();
    int status;

    assert(child >= 0);
    if (child == 0) {
        alarm(5);
        hostpid_fallback_lock_after_fork();
        hostpid_fallback_lock_set_before_flock_hook(NULL);
        if (hostpid_fallback_lock_acquire_at(lock_path, getuid(), 1000) != 0 ||
            hostpid_fallback_lock_release() != 0) {
            _exit(2);
        }
        _exit(0);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(fcntl(acquiring_fd, F_GETFD) >= 0);
    resume_thread();
    assert(pthread_join(thread, NULL) == 0);
    assert(acquire_result == 0);
    assert(hostpid_fallback_lock_release() == 0);
}

int main(void) {
    char directory[] = "/tmp/hami-hostpid-lock-threads.XXXXXX";

    alarm(15);
    lock_path = mkdtemp(directory);
    assert(lock_path != NULL);
    test_release_cannot_close_an_inflight_acquire();
    test_cancelled_acquire_does_not_strand_the_guard();
    test_fork_child_clears_an_inherited_busy_guard();
    hostpid_fallback_lock_set_before_flock_hook(NULL);
    assert(rmdir(lock_path) == 0);
    alarm(0);
    puts("host PID fallback lock thread tests passed");
    return 0;
}

#ifndef __UTILS_PROCESS_UTILS_H__
#define __UTILS_PROCESS_UTILS_H__ 

#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>

#define PROC_STATE_ALIVE 0
#define PROC_STATE_NONALIVE 1
#define PROC_STATE_UNKNOWN 2

/*
 * Headers older than Linux 5.3, such as the UBI8 ones the release image is
 * built on, lack the number.  It is 434 on every architecture HAMi builds
 * for; a kernel without it returns ENOSYS and proc_alive falls back to /proc.
 */
#if !defined(SYS_pidfd_open) && (defined(__x86_64__) || defined(__aarch64__))
#define SYS_pidfd_open 434
#endif

#if defined(SYS_pidfd_open) && !defined(PROC_ALIVE_NO_PIDFD)
/*
 * A pidfd polls readable once the leader has exited and no other thread is
 * left, which is exactly when the process is gone.  Returns -1 when pidfds
 * are unavailable (kernel before 5.3, or a seccomp profile that denies it).
 */
static int proc_alive_pidfd(int32_t pid) {
    struct pollfd pfd = {.events = POLLIN};
    int ready;

    pfd.fd = syscall(SYS_pidfd_open, pid, 0);
    if (pfd.fd < 0) {
        return errno == ESRCH ? PROC_STATE_NONALIVE : -1;
    }
    do {
        ready = poll(&pfd, 1, 0);
    } while (ready < 0 && errno == EINTR);
    close(pfd.fd);
    if (ready < 0) {
        return -1;
    }
    return ready > 0 ? PROC_STATE_NONALIVE : PROC_STATE_ALIVE;
}
#endif

/*
 * Without pidfds: kill() finds the process until it is reaped, and /proc
 * tells an unreaped zombie apart.  Only a zombie leader with no thread left
 * is gone; the leader turns Z as soon as its own thread exits, while the
 * other threads still hold the process's files, and with them its GPU
 * memory.  A /proc entry that cannot be read is UNKNOWN, never NONALIVE.
 */
static int proc_alive_procfs(int32_t pid) {
    char path[32];
    char buf[512];
    int fd;
    ssize_t n;
    char *comm_end;
    char state;
    int64_t threads;

    if (kill(pid, 0) != 0 && errno == ESRCH) {
        return PROC_STATE_NONALIVE;
    }
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return PROC_STATE_UNKNOWN;
    }
    do {
        n = read(fd, buf, sizeof(buf) - 1);
    } while (n < 0 && errno == EINTR);
    close(fd);
    if (n <= 0) {
        return PROC_STATE_UNKNOWN;
    }
    buf[n] = '\0';
    /* comm may hold spaces and parentheses; it ends at the last ')'. */
    comm_end = strrchr(buf, ')');
    /* state is field 3, num_threads field 20 */
    if (comm_end == NULL ||
        sscanf(comm_end + 1,
               " %c %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s"
               " %*s %*s %" SCNd64, &state, &threads) != 2) {
        return PROC_STATE_UNKNOWN;
    }
    if (state == 'Z' || state == 'X' || state == 'x') {
        return threads <= 1 ? PROC_STATE_NONALIVE : PROC_STATE_ALIVE;
    }
    return PROC_STATE_ALIVE;
}

int proc_alive(int32_t pid) {
    int saved_errno = errno;
    int res = -1;

    if (pid <= 0) {
        return PROC_STATE_NONALIVE;
    }
#if defined(SYS_pidfd_open) && !defined(PROC_ALIVE_NO_PIDFD)
    res = proc_alive_pidfd(pid);
#endif
    if (res < 0) {
        res = proc_alive_procfs(pid);
    }
    errno = saved_errno;
    return res;
}


#endif  // __UTILS_PROCESS_UTILS_H__

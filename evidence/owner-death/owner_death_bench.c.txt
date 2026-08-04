#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define FILE_PATH "/tmp/hami-owner-death.lock"

static long long now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

static void sleep_us(unsigned int us)
{
    struct timespec ts = {
        .tv_sec = us / 1000000U,
        .tv_nsec = (long)(us % 1000000U) * 1000L
    };

    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {}
}

/* Mirrors #248 */
static void hami_backoff(unsigned int backoff_us, uint32_t *state)
{
    unsigned int minimum_us = backoff_us / 2;
    unsigned int range_us = backoff_us - minimum_us;

    *state = *state * UINT32_C(1103515245) + UINT32_C(12345);

    unsigned int delay_us =
        minimum_us + *state % (range_us + 1);

    sleep_us(delay_us);
}

static uint32_t hami_seed(void)
{
    struct timespec now;
    uint32_t seed = (uint32_t)getpid();

    if (clock_gettime(CLOCK_MONOTONIC, &now) == 0) {
        seed ^= (uint32_t)now.tv_sec;
        seed ^= (uint32_t)now.tv_nsec;
    }

    return seed;
}

static int acquire_poll(int fd)
{
    struct flock lock = {
        .l_type = F_WRLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 1
    };

    unsigned int delay = 10000U;
    uint32_t jitter = hami_seed();

    for (;;) {
        int rc;

        do {
            rc = fcntl(fd, F_SETLK, &lock);
        } while (rc != 0 && errno == EINTR);

        if (rc == 0)
            return 0;

        if (errno != EACCES && errno != EAGAIN)
            return -1;

        hami_backoff(delay, &jitter);

        if (delay < 500000U)
            delay *= 2;
        else
            delay = 1000000U;
    }
}

static int acquire_block(int fd)
{
    struct flock lock = {
        .l_type = F_WRLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 1
    };

    int rc;

    do {
        rc = fcntl(fd, F_SETLKW, &lock);
    } while (rc != 0 && errno == EINTR);

    return rc;
}

static int lock_now(int fd)
{
    struct flock lock = {
        .l_type = F_WRLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 1
    };

    return fcntl(fd, F_SETLK, &lock);
}

int main(int argc, char **argv)
{
    if (argc != 2 ||
        (strcmp(argv[1], "poll") && strcmp(argv[1], "block"))) {
        fprintf(stderr, "usage: %s poll|block\n", argv[0]);
        return 2;
    }

    int fd = open(FILE_PATH, O_CREAT | O_RDWR, 0600);

    if (fd < 0) {
        perror("open");
        return 1;
    }

    int ready[2];
    pipe(ready);

    pid_t owner = fork();

    if (owner == 0) {
        close(ready[0]);

        if (lock_now(fd) != 0)
            _exit(10);

        write(ready[1], "R", 1);

        /* Must die while owning the record lock */
        for (;;)
            pause();
    }

    close(ready[1]);

    char c;
    read(ready[0], &c, 1);
    close(ready[0]);

    int result[2];
    pipe(result);

    pid_t waiter = fork();

    if (waiter == 0) {
        close(result[0]);

        long long before = now_us();

        int rc = !strcmp(argv[1], "poll")
            ? acquire_poll(fd)
            : acquire_block(fd);

        long long acquired = now_us();

        if (rc != 0)
            _exit(11);

        write(result[1], &acquired, sizeof(acquired));
        close(result[1]);
        _exit(0);
    }

    close(result[1]);

    /*
     * Give waiter 100 ms to reach F_SETLKW or the polling loop.
     */
    usleep(100000);

    long long killed = now_us();

    kill(owner, SIGKILL);
    waitpid(owner, NULL, 0);

    long long acquired;

    if (read(result[0], &acquired, sizeof(acquired)) != sizeof(acquired)) {
        fprintf(stderr, "failed to receive waiter result\n");
        kill(waiter, SIGKILL);
        return 1;
    }

    waitpid(waiter, NULL, 0);

    printf("mode=%s recovery_us=%lld recovery_ms=%.3f\n",
           argv[1],
           acquired - killed,
           (acquired - killed) / 1000.0);

    close(fd);
    unlink(FILE_PATH);

    return 0;
}

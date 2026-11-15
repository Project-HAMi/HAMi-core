/*
 * GPU-free regression: a forked child re-runs try_create_shrreg() and must
 * release the shared-region fd and mapping it inherited, at any fork depth.
 */
#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "multiprocess/multiprocess_memory_limit.h"
static int count_fds(const char *path) {
    DIR *dir = opendir("/proc/self/fd");
    struct dirent *e;
    char link[PATH_MAX], target[PATH_MAX];
    int n = 0;
    while (dir != NULL && (e = readdir(dir)) != NULL) {
        snprintf(link, sizeof(link), "/proc/self/fd/%s", e->d_name);
        ssize_t len = readlink(link, target, sizeof(target) - 1);
        target[len > 0 ? len : 0] = '\0';
        n += len > 0 && strcmp(target, path) == 0;
    }
    if (dir != NULL) {
        closedir(dir);
    }
    return n;
}
static int count_maps(const char *path) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[PATH_MAX + 256];
    int n = 0;
    while (f != NULL && fgets(line, sizeof(line), f) != NULL) {
        n += strstr(line, path) != NULL;
    }
    if (f != NULL) {
        fclose(f);
    }
    return n;
}
int main(void) {
    char path[] = "/tmp/hami-fork-reinit.XXXXXX";
    int fd = mkstemp(path), depth, status = 0;
    if (fd < 0 || setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, path, 1) != 0) {
        return 1;
    }
    close(fd);
    setenv("LIBCUDA_LOG_LEVEL", "0", 1);
    log_utils_init();
    ensure_initialized();
    for (depth = 0; depth <= 3; depth++) {
        if (count_fds(path) != 1 || count_maps(path) != 1) {
            fprintf(stderr, "depth %d: %d fds, %d mappings, want 1 and 1\n",
                    depth, count_fds(path), count_maps(path));
            status = 1;
            break;
        }
        pid_t child = fork();
        if (child > 0) {
            waitpid(child, &status, 0);
            status = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
            break;
        }
        if (child < 0) {
            status = 1;
            break;
        }
        ensure_initialized();
    }
    if (depth == 0) {
        unlink(path);
        puts(status == 0 ? "fork reinit leak tests passed" : "fork reinit leak tests failed");
    }
    return status;
}

/*
 * GPU-free regression test for the override file.
 *
 * A process started from SSH, su -, sudo or cron has none of the limit
 * variables. The file has to be loaded before the shared region is created,
 * otherwise the region is stamped with limit 0, which means no limit, in a
 * private cache. One bad line in the file must not drop the lines after it.
 *
 * Each case runs in its own child because the library initializes once per
 * process.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define MiB (UINT64_C(1) << 20)

static const char *const limit_env[] = {
    "CUDA_DEVICE_MEMORY_LIMIT",
    "CUDA_DEVICE_MEMORY_LIMIT_0",
    "CUDA_DEVICE_SM_LIMIT",
    "CUDA_DEVICE_SM_LIMIT_0",
    MULTIPROCESS_SHARED_REGION_CACHE_ENV,
    MULTIPROCESS_SHARED_REGION_HOOK_PATH_ENV,
    MULTIPROCESS_SHARED_REGION_POD_UID_ENV,
    MULTIPROCESS_SHARED_REGION_CONTAINER_NAME_ENV,
};

/* A unique path that does not exist yet, for a shared region cache. */
static void fresh_cache(char *path, size_t len) {
    int fd;

    snprintf(path, len, "/tmp/hami-override-env.XXXXXX");
    fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        _exit(2);
    }
    close(fd);
    unlink(path);
}

static void write_override(const char *content) {
    FILE *f = fopen(ENV_OVERRIDE_FILE, "w");

    if (f == NULL || fputs(content, f) == EOF || fclose(f) != 0) {
        perror("write " ENV_OVERRIDE_FILE);
        _exit(2);
    }
}

/* Initialize, then compare what the library holds with what was asked. */
static int expect(uint64_t limit, int sm, const char *cache) {
    struct stat st;
    uint64_t got_limit;
    int got_sm;
    int rc = 0;

    ensure_initialized();
    got_limit = get_current_device_memory_limit(0);
    got_sm = get_current_device_sm_limit(0);
    if (got_limit != limit) {
        fprintf(stderr, "memory limit %" PRIu64 ", expected %" PRIu64 "\n",
                got_limit, limit);
        rc = 1;
    }
    if (sm >= 0 && got_sm != sm) {
        fprintf(stderr, "sm limit %d, expected %d\n", got_sm, sm);
        rc = 1;
    }
    /* The region must be the one named in the file, not a private one. */
    if (stat(cache, &st) != 0 || st.st_size == 0) {
        fprintf(stderr, "no shared region at %s\n", cache);
        rc = 1;
    }
    unlink(cache);
    return rc;
}

/* The login-session case: no variables at all, only the file. */
static int file_only(void) {
    char cache[64];
    char content[256];

    fresh_cache(cache, sizeof(cache));
    snprintf(content, sizeof(content),
             "CUDA_DEVICE_MEMORY_LIMIT_0=2048m\n"
             "CUDA_DEVICE_MEMORY_SHARED_CACHE=%s\n"
             "CUDA_DEVICE_SM_LIMIT=50\n",
             cache);
    write_override(content);
    return expect(2048 * MiB, 50, cache);
}

/* When both are present the file is the source of truth. */
static int file_beats_env(void) {
    char env_cache[64];
    char cache[64];
    char content[256];
    int rc;

    fresh_cache(env_cache, sizeof(env_cache));
    fresh_cache(cache, sizeof(cache));
    setenv("CUDA_DEVICE_MEMORY_LIMIT_0", "512m", 1);
    setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, env_cache, 1);
    snprintf(content, sizeof(content),
             "CUDA_DEVICE_MEMORY_LIMIT_0=2048m\n"
             "CUDA_DEVICE_MEMORY_SHARED_CACHE=%s\n",
             cache);
    write_override(content);
    rc = expect(2048 * MiB, -1, cache);
    unlink(env_cache);
    return rc;
}

/* A hand-edited file: a blank line, a line with no '=', a CRLF ending.
   Everything after the bad line must still be loaded. */
static int bad_line(void) {
    char cache[64];
    char content[256];

    fresh_cache(cache, sizeof(cache));
    snprintf(content, sizeof(content),
             "CUDA_DEVICE_SM_LIMIT=30\n"
             "\n"
             "this line has no equals sign\n"
             "CUDA_DEVICE_MEMORY_LIMIT_0=1024m\n"
             "CUDA_DEVICE_MEMORY_SHARED_CACHE=%s\r\n",
             cache);
    write_override(content);
    return expect(1024 * MiB, 30, cache);
}

/* No file: the environment works exactly as before. */
static int no_file(void) {
    char cache[64];

    fresh_cache(cache, sizeof(cache));
    unlink(ENV_OVERRIDE_FILE);
    setenv("CUDA_DEVICE_MEMORY_LIMIT_0", "512m", 1);
    setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, cache, 1);
    return expect(512 * MiB, -1, cache);
}

int main(void) {
    static const struct {
        const char *name;
        int (*run)(void);
    } cases[] = {
        {"file only", file_only},
        {"file beats environment", file_beats_env},
        {"bad line in the middle", bad_line},
        {"no file", no_file},
    };
    int failures = 0;
    size_t i;

    /* Never let this test write the production path. */
    if (strcmp(ENV_OVERRIDE_FILE, "/overrideEnv") == 0) {
        fprintf(stderr, "build with -DENV_OVERRIDE_FILE=<scratch path>\n");
        return 1;
    }
    for (i = 0; i < sizeof(limit_env) / sizeof(limit_env[0]); i++) {
        unsetenv(limit_env[i]);
    }
    log_utils_init();

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        int status;
        pid_t pid = fork();

        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            _exit(cases[i].run());
        }
        if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
            WEXITSTATUS(status) != 0) {
            fprintf(stderr, "case \"%s\" failed\n", cases[i].name);
            failures++;
        }
    }
    unlink(ENV_OVERRIDE_FILE);

    if (failures != 0) {
        fprintf(stderr, "%d override env case(s) failed\n", failures);
        return 1;
    }
    puts("override env tests passed");
    return 0;
}

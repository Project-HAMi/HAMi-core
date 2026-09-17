/*
 * GPU-free regression test for early /overrideEnv loading.
 *
 * HAMi-core must load ENV_OVERRIDE_FILE before shared-region creation so that
 * env-stripped login/SSH processes recover their allocation limits before
 * try_create_shrreg() reads CUDA_DEVICE_MEMORY_SHARED_CACHE and
 * CUDA_DEVICE_MEMORY_LIMIT_*.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

static int write_override_env(const char *cache_path) {
    FILE *f = fopen(ENV_OVERRIDE_FILE, "w");
    if (f == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", ENV_OVERRIDE_FILE, strerror(errno));
        return -1;
    }

    if (fprintf(f,
                "CUDA_DEVICE_MEMORY_LIMIT_0=2048m\n"
                "CUDA_DEVICE_SM_LIMIT=50\n"
                "CUDA_DEVICE_MEMORY_SHARED_CACHE=%s\n",
                cache_path) < 0) {
        fprintf(stderr, "failed to write %s\n", ENV_OVERRIDE_FILE);
        fclose(f);
        return -1;
    }

    if (fclose(f) != 0) {
        fprintf(stderr, "failed to close %s: %s\n", ENV_OVERRIDE_FILE, strerror(errno));
        return -1;
    }

    return 0;
}

int main(void) {
    char cache_template[] = "/tmp/hami-override-env-cache-XXXXXX";
    int cache_fd = mkstemp(cache_template);
    if (cache_fd < 0) {
        perror("mkstemp(cache)");
        return 1;
    }
    close(cache_fd);
    unlink(cache_template);

    unlink(ENV_OVERRIDE_FILE);

    unsetenv("CUDA_DEVICE_MEMORY_LIMIT");
    unsetenv("CUDA_DEVICE_MEMORY_LIMIT_0");
    unsetenv("CUDA_DEVICE_SM_LIMIT");
    unsetenv("CUDA_DEVICE_SM_LIMIT_0");
    unsetenv("CUDA_DEVICE_MEMORY_SHARED_CACHE");

    if (write_override_env(cache_template) != 0) {
        return 1;
    }

    ensure_initialized();

    const char *loaded_limit = getenv("CUDA_DEVICE_MEMORY_LIMIT_0");
    const char *loaded_sm = getenv("CUDA_DEVICE_SM_LIMIT");
    const char *loaded_cache = getenv("CUDA_DEVICE_MEMORY_SHARED_CACHE");

    if (loaded_limit == NULL || strcmp(loaded_limit, "2048m") != 0) {
        fprintf(stderr, "CUDA_DEVICE_MEMORY_LIMIT_0 was not loaded early\n");
        return 1;
    }
    if (loaded_sm == NULL || strcmp(loaded_sm, "50") != 0) {
        fprintf(stderr, "CUDA_DEVICE_SM_LIMIT was not loaded early\n");
        return 1;
    }
    if (loaded_cache == NULL || strcmp(loaded_cache, cache_template) != 0) {
        fprintf(stderr, "CUDA_DEVICE_MEMORY_SHARED_CACHE was not loaded early\n");
        return 1;
    }

    uint64_t limit = get_current_device_memory_limit(0);
    uint64_t expected_limit = 2048ULL * 1024ULL * 1024ULL;
    if (limit != expected_limit) {
        fprintf(stderr, "device 0 memory limit = %" PRIu64 ", expected %" PRIu64 "\n",
                limit, expected_limit);
        return 1;
    }

    if (get_current_device_sm_limit(0) != 50) {
        fprintf(stderr, "device 0 SM limit was not initialized from override env\n");
        return 1;
    }

    if (access(cache_template, F_OK) != 0) {
        fprintf(stderr, "shared-region cache was not created at override path: %s\n",
                cache_template);
        return 1;
    }

    unlink(cache_template);
    unlink(ENV_OVERRIDE_FILE);
    return 0;
}

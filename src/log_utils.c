#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

/*
 * Cached log level, read once from LIBCUDA_LOG_LEVEL by log_utils_init().
 * Default 2 = warn/msg/error (matches original behavior when env is unset).
 */
int g_log_level = 2;

FILE *fp1 = NULL;

static pthread_once_t log_level_once = PTHREAD_ONCE_INIT;

void log_utils_init(void) {
    const char *env = getenv("LIBCUDA_LOG_LEVEL");
    if (env != NULL) {
        g_log_level = atoi(env);
    }
    /* else: keep default of 2 (warn level) */
}

/*
 * The dlsym hook and preInit() log before preInit() reaches log_utils_init(),
 * so reading the cached value directly ignores LIBCUDA_LOG_LEVEL for exactly
 * those messages. Resolve on first use instead.
 */
int log_level(void) {
    pthread_once(&log_level_once, log_utils_init);
    return g_log_level;
}

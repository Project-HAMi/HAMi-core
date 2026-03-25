#include <stdio.h>
#include <stdlib.h>

/*
 * Cached log level, read once from LIBCUDA_LOG_LEVEL by log_utils_init().
 * Default 2 = warn/msg/error (matches original behavior when env is unset).
 */
int g_log_level = 2;

FILE *fp1 = NULL;

void log_utils_init(void) {
    const char *env = getenv("LIBCUDA_LOG_LEVEL");
    if (env != NULL) {
        g_log_level = atoi(env);
    }
    /* else: keep default of 2 (warn level) */
}

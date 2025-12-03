// Config file reader for softmig
// Reads secure config from /var/run/softmig/{jobid}_{arrayid}.conf
// Falls back to environment variables if config file doesn't exist

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include "include/log_utils.h"

// Get config file path for current SLURM job
static char* get_config_file_path(void) {
    static char config_path[512] = {0};
    static int initialized = 0;
    
    if (initialized) {
        return config_path;
    }
    
    char* job_id = getenv("SLURM_JOB_ID");
    char* array_id = getenv("SLURM_ARRAY_TASK_ID");
    
    if (job_id == NULL) {
        // Not in SLURM job - no config file
        config_path[0] = '\0';
        initialized = 1;
        return config_path;
    }
    
    // Build path: /var/run/softmig/{jobid}_{arrayid}.conf or /var/run/softmig/{jobid}.conf
    if (array_id != NULL && strlen(array_id) > 0) {
        snprintf(config_path, sizeof(config_path), "/var/run/softmig/%s_%s.conf", job_id, array_id);
    } else {
        snprintf(config_path, sizeof(config_path), "/var/run/softmig/%s.conf", job_id);
    }
    
    initialized = 1;
    return config_path;
}

// Read a value from config file (key=value format)
// Returns 1 if found, 0 if not found
static int read_config_value(const char* key, char* value, size_t value_size) {
    char* config_path = get_config_file_path();
    if (config_path[0] == '\0') {
        return 0;  // No config file
    }
    
    FILE* f = fopen(config_path, "r");
    if (f == NULL) {
        return 0;  // Config file doesn't exist
    }
    
    char line[1024];
    size_t key_len = strlen(key);
    
    while (fgets(line, sizeof(line), f) != NULL) {
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\0') {
            continue;
        }
        
        // Check if this line starts with our key
        if (strncmp(line, key, key_len) == 0 && line[key_len] == '=') {
            // Found it - copy value
            strncpy(value, line + key_len + 1, value_size - 1);
            value[value_size - 1] = '\0';
            fclose(f);
            return 1;
        }
    }
    
    fclose(f);
    return 0;  // Key not found in config file
}

// Get limit from config file or environment variable
// Priority: config file > environment variable
size_t get_limit_from_config_or_env(const char* env_name) {
    char config_value[256] = {0};
    
    // Try to read from config file first (if in SLURM job)
    if (read_config_value(env_name, config_value, sizeof(config_value))) {
        // Parse the config value (same format as env var: "16g", "24G", etc.)
        size_t len = strlen(config_value);
        if (len == 0) {
            return 0;
        }
        
        size_t scalar = 1;
        char* digit_end = config_value + len;
        if (config_value[len - 1] == 'G' || config_value[len - 1] == 'g') {
            digit_end -= 1;
            scalar = 1024 * 1024 * 1024;
        } else if (config_value[len - 1] == 'M' || config_value[len - 1] == 'm') {
            digit_end -= 1;
            scalar = 1024 * 1024;
        } else if (config_value[len - 1] == 'K' || config_value[len - 1] == 'k') {
            digit_end -= 1;
            scalar = 1024;
        }
        
        size_t res = strtoul(config_value, &digit_end, 0);
        size_t scaled_res = res * scalar;
        
        if (scaled_res == 0) {
            if (strstr(env_name, "SM_LIMIT") != NULL) {
                LOG_INFO("device core util limit set to 0 from config, which means no limit: %s=%s",
                    env_name, config_value);
            } else {
                LOG_WARN("invalid device memory limit from config %s=%s", env_name, config_value);
            }
            return 0;
        }
        
        if (scaled_res != 0 && scaled_res / scalar != res) {
            LOG_ERROR("Limit overflow from config: %s=%s", env_name, config_value);
            return 0;
        }
        
        LOG_DEBUG("Read %s=%s from config file", env_name, config_value);
        return scaled_res;
    }
    
    // Fallback to environment variable (for non-SLURM or if config file doesn't exist)
    char* env_limit = getenv(env_name);
    if (env_limit == NULL) {
        return 0;
    }
    
    size_t len = strlen(env_limit);
    if (len == 0) {
        return 0;
    }
    
    size_t scalar = 1;
    char* digit_end = env_limit + len;
    if (env_limit[len - 1] == 'G' || env_limit[len - 1] == 'g') {
        digit_end -= 1;
        scalar = 1024 * 1024 * 1024;
    } else if (env_limit[len - 1] == 'M' || env_limit[len - 1] == 'm') {
        digit_end -= 1;
        scalar = 1024 * 1024;
    } else if (env_limit[len - 1] == 'K' || env_limit[len - 1] == 'k') {
        digit_end -= 1;
        scalar = 1024;
    }
    
    size_t res = strtoul(env_limit, &digit_end, 0);
    size_t scaled_res = res * scalar;
    
    if (scaled_res == 0) {
        if (strstr(env_name, "SM_LIMIT") != NULL) {
            LOG_INFO("device core util limit set to 0, which means no limit: %s=%s",
                env_name, env_limit);
        } else if (strstr(env_name, "MEMORY_LIMIT") != NULL) {
            LOG_WARN("invalid device memory limit %s=%s", env_name, env_limit);
        } else {
            LOG_WARN("invalid env name:%s", env_name);
        }
        return 0;
    }
    
    if (scaled_res != 0 && scaled_res / scalar != res) {
        LOG_ERROR("Limit overflow: %s=%s", env_name, env_limit);
        return 0;
    }
    
    return scaled_res;
}

// Delete config file on job exit
void cleanup_config_file(void) {
    char* config_path = get_config_file_path();
    if (config_path[0] != '\0') {
        if (unlink(config_path) == 0) {
            LOG_DEBUG("Deleted config file: %s", config_path);
        } else {
            LOG_DEBUG("Could not delete config file %s (may not exist): %d", config_path, errno);
        }
    }
}

// Check if softmig should be active (either CUDA_DEVICE_MEMORY_LIMIT or CUDA_DEVICE_SM_LIMIT is set)
// Returns 1 if at least one is set, 0 if neither is set
int is_softmig_configured(void) {
    // Check if CUDA_DEVICE_MEMORY_LIMIT is set (from config file or environment)
    size_t memory_limit = get_limit_from_config_or_env("CUDA_DEVICE_MEMORY_LIMIT");
    if (memory_limit > 0) {
        return 1;
    }
    
    // Check if CUDA_DEVICE_SM_LIMIT is set (from config file or environment)
    size_t sm_limit = get_limit_from_config_or_env("CUDA_DEVICE_SM_LIMIT");
    if (sm_limit > 0) {
        return 1;
    }
    
    // Neither is set - softmig should be passive
    return 0;
}


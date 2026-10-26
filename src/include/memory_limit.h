#ifndef __MEMORY_LIMIT_H__
#define __MEMORY_LIMIT_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#include "static_config.h"


#define CUDA_DEVICE_MEMORY_LIMIT "CUDA_DEVICE_MEMORY_LIMIT"
#define CUDA_DEVICE_MEMORY_LIMIT_KEY_LENGTH 32
#define CUDA_DEVICE_SM_LIMIT "CUDA_DEVICE_SM_LIMIT"
#define CUDA_DEVICE_SM_LIMIT_KEY_LENGTH 32

#define ENSURE_INITIALIZED() ensure_initialized();        \

#define ENSURE_RUNNING() {                                \
    ensure_initialized();                                 \
    if (is_gpu_core_limit_enabled()) {                    \
        while(!wait_status_self(1)) { LOG_DEBUG("E1"); sleep(1); }         \
    }                                                      \
}                                                         \

#include "multiprocess/multiprocess_memory_limit.h"

#endif  // __MEMORY_LIMIT_H__

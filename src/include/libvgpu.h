#ifndef __LIBVGPU_H__
#define __LIBVGPU_H__

#include <dlfcn.h>
#include <cuda.h>
#include "include/nvml_prefix.h"
#include <nvml.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <signal.h>

#include "include/log_utils.h"
#include "static_config.h"

#define ENSURE_INITIALIZED() ensure_initialized();        \

extern void load_cuda_libraries();

#if defined(__GNUC__) && defined(__GLIBC__)

#define FUNC_ATTR_VISIBLE  __attribute__((visibility("default")))

#if defined(DLSYM_HOOK_DEBUG)
#define DLSYM_HOOK_FUNC(f)                                       \
    if (0 == strcmp(symbol, #f)) {                               \
        LOG_DEBUG("Detect dlsym for %s\n", #f);                  \
        return (void*) f; }                                      \

#else 

#define DLSYM_HOOK_FUNC(f)                                       \
    if (0 == strcmp(symbol, #f)) {                               \
        return (void*) f; }                                      \

#endif     

void* __dlsym_hook_section(void* handle, const char* symbol);
void* __dlsym_hook_section_nvml(void* handle, const char* symbol);

typedef void* (*fp_dlsym)(void*, const char*);

#else
#error error, neither __GLIBC__ nor __GNUC__ defined

#endif

nvmlReturn_t set_task_pid();
// UNKNOWN means the broker query failed; NOT_FOUND means no shared PID slot.
nvmlReturn_t set_task_pid_from_broker(void);
void size_primary_context_lazily(int cudadev);
void reset_lazy_context_sizing(void);
int map_cuda_visible_devices();
void ensure_post_init();

#endif  // SRC_INCLUDE_LIBVGPU_H_

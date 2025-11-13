#ifndef __LIBSOFTMIG_H__
#define __LIBSOFTMIG_H__

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
//#include "memory_limit.h"

#define ENSURE_INITIALIZED() ensure_initialized();        \

extern void load_cuda_libraries();

#if defined(__GNUC__) && defined(__GLIBC__)

#define FUNC_ATTR_VISIBLE  __attribute__((visibility("default"))) 
#define FUNC_PTR_TYPE(fname) __func_ptr_type_##fname
#define FUNC_PTR_NAME(fname) __func_ptr_origin_##fname
#define FUNC_PTR_ALIAS_ATTR(overrided)                           \
        __attribute__((alias(#overrided), used))                 \
        FUNC_ATTR_VISIBLE;                                       \

#define FUNC_OVERRIDE_NAME(fname) overrided_##fname

// _dl_sym is an internal glibc function, use weak linking if available
#ifdef __GLIBC__
extern void* _dl_sym(void*, const char*, void*) __attribute__((weak));
#endif

#if defined(DLSYM_HOOK_DEBUG)
#define DLSYM_HOOK_FUNC(f)                                       \
    if (0 == strcmp(symbol, #f)) {                               \
        LOG_DEBUG("Detect dlsym for %s\n", #f);                  \
        return (void*) f; }                                      \

#define DLSYM_HOOK_FUNC_REPLACE(f)                               \
    if (0 == strcmp(symbol, hacked_#f)) {                        \
        return (void*) f; }                                      \

#else 

#define DLSYM_HOOK_FUNC(f)                                       \
    if (0 == strcmp(symbol, #f)) {                               \
        return (void*) f; }                                      \

#define DLSYM_HOOK_FUNC_REPLACE(f)                               \
    if (0 == strcmp(symbol, #f)) {                        \
        return (void*) hacked_##f; }                                      \

#endif     

void* __dlsym_hook_section(void* handle, const char* symbol);
void* __dlsym_hook_section_nvml(void* handle, const char* symbol);

typedef void* (*fp_dlsym)(void*, const char*);

#else
#error error, neither __GLIBC__ nor __GNUC__ defined

#endif

/* Determine the return address.  */
#define RETURN_ADDRESS(nr) \
  __builtin_extract_return_addr (__builtin_return_address (nr))

nvmlReturn_t set_task_pid();
int map_cuda_visible_devices();

#endif  // __LIBSOFTMIG_H__
#define _GNU_SOURCE

#include <dlfcn.h>
#include <stddef.h>

/* Matches the weak forwarding symbol exported by LLVM OpenMP. */
__attribute__((weak, visibility("default"), noinline, used))
void* ompt_start_tool(unsigned int omp_version, const char* runtime_version) {
    (void)omp_version;
    (void)runtime_version;
    return NULL;
}

void* lookup_default_ompt(void) {
    void* volatile result = dlsym(RTLD_DEFAULT, "ompt_start_tool");
    return result;
}

void* lookup_next_ompt(void) {
    void* volatile result = dlsym(RTLD_NEXT, "ompt_start_tool");
    return result;
}

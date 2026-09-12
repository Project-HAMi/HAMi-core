#define _GNU_SOURCE

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

/*
 * Repeated RTLD_NEXT lookups must return the same address. The old
 * (thread, pointer) history map treated the second call as recursion
 * and replaced a valid pointer with NULL.
 */
int main(void) {
    void *first = dlsym(RTLD_NEXT, "open");
    void *second = dlsym(RTLD_NEXT, "open");

    if (first == NULL) {
        fprintf(stderr, "first RTLD_NEXT open lookup failed: %s\n", dlerror());
        return 1;
    }
    if (second == NULL) {
        fprintf(stderr, "second RTLD_NEXT open lookup returned NULL\n");
        return 1;
    }
    if (first != second) {
        fprintf(stderr, "RTLD_NEXT open lookup changed: %p then %p\n",
                first, second);
        return 1;
    }

    /*
     * CUDA drivers can resolve their own entry points through dlsym(handle,
     * ...). The preload hook must preserve that handle when no redirect
     * library is configured, otherwise the lookup can recurse into HAMi.
     */
    void *cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (cuda == NULL) {
        printf("SKIP libcuda.so.1 unavailable: %s\n", dlerror());
        return 77;
    }
    dlerror();
    void *cu_init = dlsym(cuda, "cuInit");
    const char *error = dlerror();
    if (cu_init == NULL || error != NULL) {
        fprintf(stderr, "libcuda cuInit lookup failed: %s\n",
                error != NULL ? error : "NULL symbol");
        dlclose(cuda);
        return 1;
    }
    Dl_info info = {0};
    if (dladdr(cu_init, &info) == 0 || info.dli_fname == NULL ||
            strstr(info.dli_fname, "libvgpu.so") != NULL) {
        fprintf(stderr, "libcuda cuInit resolved to an unexpected library: %s\n",
                info.dli_fname != NULL ? info.dli_fname : "unknown");
        dlclose(cuda);
        return 1;
    }
    dlclose(cuda);
    printf("PASS %p\n", first);
    return 0;
}

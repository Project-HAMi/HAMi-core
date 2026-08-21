#define _GNU_SOURCE

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>

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
    printf("PASS %p\n", first);
    return 0;
}

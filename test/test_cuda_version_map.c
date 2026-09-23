/*
 * libcudart asks cuGetProcAddress for a base name and expects the spelling
 * cuda.h gave it for its own CUDA version. find_symbols_in_table() probes
 * _v3, _v2 and the bare name instead, so g_func_map has to pin the rest.
 *
 * The stub below exports exactly what a build of that version compiles in,
 * which is what makes the pass-through cases real rather than incidental.
 */
#include <stdio.h>
#include <string.h>

#include "include/libcuda_hook.h"

void *find_symbols_in_table_by_cudaversion(const char *symbol, int cudaVersion);
CUresult _cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                              cuuint64_t flags,
                              CUdriverProcAddressQueryResult *symbolStatus);

/* Referenced by load_cuda_libraries(), which this test never calls. */
typedef void *(*fp_dlsym)(void *, const char *);
fp_dlsym real_dlsym;

/* Set per case to the hook spellings the build under test exports. */
static const char *g_present[4];

void *__dlsym_hook_section(void *handle, const char *symbol) {
    (void)handle;
    for (int i = 0; g_present[i] != NULL; i++) {
        if (strcmp(g_present[i], symbol) == 0) {
            return (void *)g_present[i]; /* identifies the winner, never called */
        }
    }
    return NULL;
}

static int check(const char *what, const char *base, const char *const *present,
                 int cudaVersion, const char *expect) {
    for (int i = 0; i < 4; i++)
        g_present[i] = (i < 3) ? present[i] : NULL;

    void *pfn = find_symbols_in_table_by_cudaversion(base, cudaVersion);
    const char *got = pfn ? (const char *)pfn : "(pass-through)";
    const char *want = expect ? expect : "(pass-through)";

    if (strcmp(got, want) != 0) {
        printf("FAIL %s %s: cudaVersion=%d resolved to %s, expected %s\n",
               base, what, cudaVersion, got, want);
        return 1;
    }
    printf("ok   %s %s: cudaVersion=%d -> %s\n", base, what, cudaVersion, got);
    return 0;
}

static int check_pair(const char *base, const char *const *cuda12,
                      const char *const *cuda13, const char *on12,
                      const char *on13) {
    int failures = 0;
    failures += check("cuda12 build, cuda12 app", base, cuda12, 12090, on12);
    failures += check("cuda12 build, cuda13 app", base, cuda12, 13030, on13);
    failures += check("cuda13 build, cuda13 app", base, cuda13, 13030, on13);
    /* The 12.x spelling is compiled out of a CUDA 13 build. */
    failures += check("cuda13 build, cuda12 app", base, cuda13, 12090, NULL);
    return failures;
}

/* The driver writes symbolStatus for every answered lookup, so a table hit
   has to reach it too or the caller reads whatever was already there. */
#define STATUS_POISON ((CUdriverProcAddressQueryResult)0x5A5A)

static CUdriverProcAddressQueryResult *seen_status;

static CUresult stub_get_proc_address_v2(const char *symbol, void **pfn,
                                         int cudaVersion, cuuint64_t flags,
                                         CUdriverProcAddressQueryResult *status) {
    (void)symbol; (void)cudaVersion; (void)flags;
    seen_status = status;
    if (pfn != NULL) *pfn = NULL;
    if (status != NULL) *status = CU_GET_PROC_ADDRESS_SUCCESS;
    return CUDA_SUCCESS;
}

static int check_symbol_status(void) {
    /* A name the probe already answers, so this case turns on symbolStatus
       alone and not on the version map above. */
    static const char *const present[] = {"cuMemAlloc_v2", NULL, NULL};
    CUdriverProcAddressQueryResult status = STATUS_POISON;
    void *pfn = NULL;
    CUresult res;
    int i;

    for (i = 0; i < 4; i++) g_present[i] = (i < 3) ? present[i] : NULL;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuGetProcAddress_v2)].fn_ptr =
        (void *)stub_get_proc_address_v2;
    seen_status = NULL;

    res = _cuGetProcAddress_v2("cuMemAlloc", &pfn, 13030, 0, &status);
    if (res != CUDA_SUCCESS || pfn == NULL) {
        printf("FAIL symbolStatus: table hit returned %d pfn=%p\n", res, pfn);
        return 1;
    }
    if (seen_status != &status || status != CU_GET_PROC_ADDRESS_SUCCESS) {
        printf("FAIL symbolStatus: left unwritten on the hooked path\n");
        return 1;
    }
    printf("ok   symbolStatus: written on the hooked path\n");
    return 0;
}

int main(void) {
    /* 12.x compiles in _v2, _v3, _v4; 13.x only _v4. */
    static const char *const ctx12[] = {"cuCtxCreate_v2", "cuCtxCreate_v3",
                                        "cuCtxCreate_v4"};
    static const char *const ctx13[] = {"cuCtxCreate_v4", NULL, NULL};

    /* 12.x compiles in the base name and _v2; 13.x only _v2. */
    static const char *const advise12[] = {"cuMemAdvise", "cuMemAdvise_v2", NULL};
    static const char *const advise13[] = {"cuMemAdvise_v2", NULL, NULL};

    static const char *const prefetch12[] = {"cuMemPrefetchAsync",
                                             "cuMemPrefetchAsync_v2", NULL};
    static const char *const prefetch13[] = {"cuMemPrefetchAsync_v2", NULL, NULL};

    int failures = 0;
    failures += check_pair("cuCtxCreate", ctx12, ctx13,
                           "cuCtxCreate_v2", "cuCtxCreate_v4");
    failures += check_pair("cuMemAdvise", advise12, advise13,
                           "cuMemAdvise", "cuMemAdvise_v2");
    failures += check_pair("cuMemPrefetchAsync", prefetch12, prefetch13,
                           "cuMemPrefetchAsync", "cuMemPrefetchAsync_v2");
    failures += check_symbol_status();

    if (failures) {
        printf("%d version map check(s) failed\n", failures);
        return 1;
    }
    printf("all version map checks passed\n");
    return 0;
}

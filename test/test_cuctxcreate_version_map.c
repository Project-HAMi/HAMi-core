/*
 * GPU-free regression test for cuCtxCreate hook resolution.
 *
 * cuda.h remaps the base name cuCtxCreate to a different real symbol per CUDA
 * major version: cuCtxCreate_v2 through CUDA 12.x, cuCtxCreate_v4 from CUDA
 * 13.0.  libcudart asks cuGetProcAddress for the base name and expects the
 * spelling that matches the cudaVersion it was built with.
 *
 * find_symbols_in_table() only probes symbol+"_v3", symbol+"_v2" and symbol,
 * so on its own it can never return cuCtxCreate_v4, and on CUDA 12.x it
 * answers with cuCtxCreate_v3 -- whose five-parameter signature does not match
 * the three-parameter cuCtxCreate the caller holds.  g_func_map pins both.
 *
 * The test links the production resolver from src/cuda/hook.c and supplies its
 * own __dlsym_hook_section, so it needs no driver, no NVML and no GPU.  The
 * stub answers for exactly the cuCtxCreate spellings a build of that CUDA
 * version compiles in (src/libvgpu.c guards _v2 and _v3 with
 * #if CUDA_VERSION < 13000), which is what makes the pass-through case below
 * meaningful rather than incidental.
 */
#include <stdio.h>
#include <string.h>

void *find_symbols_in_table_by_cudaversion(const char *symbol, int cudaVersion);

/* Referenced by load_cuda_libraries(), which this test never calls. */
typedef void *(*fp_dlsym)(void *, const char *);
fp_dlsym real_dlsym;

/* Set per case to the hook spellings the build under test exports. */
static const char *g_present[4];
static const char *g_asked;

void *__dlsym_hook_section(void *handle, const char *symbol) {
    (void)handle;
    for (int i = 0; g_present[i] != NULL; i++) {
        if (strcmp(g_present[i], symbol) == 0) {
            g_asked = g_present[i];
            return (void *)g_present[i]; /* identifies the winner, never called */
        }
    }
    return NULL;
}

static int check(const char *what, const char *const *present, int cudaVersion,
                 const char *expect) {
    for (int i = 0; i < 4; i++)
        g_present[i] = (i < 3 && present[i]) ? present[i] : NULL;
    g_asked = NULL;

    void *pfn = find_symbols_in_table_by_cudaversion("cuCtxCreate", cudaVersion);
    const char *got = pfn ? (const char *)pfn : "(pass-through)";
    const char *want = expect ? expect : "(pass-through)";

    if (strcmp(got, want) != 0) {
        printf("FAIL %s: cudaVersion=%d resolved to %s, expected %s\n",
               what, cudaVersion, got, want);
        return 1;
    }
    printf("ok   %s: cudaVersion=%d -> %s\n", what, cudaVersion, got);
    return 0;
}

int main(void) {
    /* A CUDA 12.x build compiles in _v2, _v3 and _v4. */
    static const char *const cuda12[] = {"cuCtxCreate_v2", "cuCtxCreate_v3",
                                         "cuCtxCreate_v4"};
    /* A CUDA 13.x build compiles in _v4 only. */
    static const char *const cuda13[] = {"cuCtxCreate_v4", NULL, NULL};

    int failures = 0;
    failures += check("cuda12 build, cuda12 app", cuda12, 12090, "cuCtxCreate_v2");
    failures += check("cuda12 build, cuda13 app", cuda12, 13030, "cuCtxCreate_v4");
    failures += check("cuda13 build, cuda13 app", cuda13, 13030, "cuCtxCreate_v4");
    /* No _v2 hook exists in a CUDA 13 build, so the legacy spelling must fall
     * through to the driver rather than reach a mismatched wrapper. */
    failures += check("cuda13 build, cuda12 app", cuda13, 12090, NULL);

    if (failures) {
        printf("%d cuCtxCreate resolution check(s) failed\n", failures);
        return 1;
    }
    printf("all cuCtxCreate resolution checks passed\n");
    return 0;
}

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
CUresult _cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);

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

typedef struct {
    int cudaVersion;
    const char *expect; /* NULL: the build compiles that spelling out */
} Case;

static int check_cases(const char *base, const char *const *present,
                       const char *build, const Case *cases, int n) {
    int failures = 0;
    for (int i = 0; i < n; i++)
        failures += check(build, base, present, cases[i].cudaVersion,
                          cases[i].expect);
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

static CUresult stream_proc_result;
static CUdriverProcAddressQueryResult stream_proc_status;
static int stream_proc_pointer, stream_proc_version;
static cuuint64_t stream_proc_flags;
static const char *stream_proc_name;
static void native_marker(void) {}

/** Return the configured native lookup result and record the request. */
static CUresult stream_get_proc(const char *symbol, void **pfn, int version, cuuint64_t flags) {
    stream_proc_name = symbol; stream_proc_version = version; stream_proc_flags = flags;
    *pfn = stream_proc_result == CUDA_SUCCESS && stream_proc_pointer ? (void *)native_marker : NULL;
    return stream_proc_result;
}

/** Return the configured native lookup result and v2 symbol status. */
static CUresult stream_get_proc_v2(const char *symbol, void **pfn, int version, cuuint64_t flags,
                                   CUdriverProcAddressQueryResult *status) {
    CUresult result = stream_get_proc(symbol, pfn, version, flags);
    if (status != NULL) *status = stream_proc_status;
    if (stream_proc_status != CU_GET_PROC_ADDRESS_SUCCESS) *pfn = NULL;
    return result;
}

/** Check stream hook selection and native lookup outcomes across both APIs. */
static int check_stream_procs(void) {
    static const char *const names[][2] = {
        {"cuMemAllocAsync", "cuMemAllocAsync_ptsz"},
        {"cuMemAllocFromPoolAsync", "cuMemAllocFromPoolAsync_ptsz"},
        {"cuMemFreeAsync", "cuMemFreeAsync_ptsz"},
        {"cuStreamSynchronize", "cuStreamSynchronize_ptsz"}
    };
    static const cuuint64_t flags[] = {CU_GET_PROC_ADDRESS_DEFAULT, CU_GET_PROC_ADDRESS_LEGACY_STREAM,
                                      CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM};
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuGetProcAddress)].fn_ptr = (void *)stream_get_proc;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuGetProcAddress_v2)].fn_ptr = (void *)stream_get_proc_v2;
    int failures = 0;
    for (int n = 0; n < 4; n++) for (int explicit_pt = 0; explicit_pt < 2; explicit_pt++)
        for (int api = 0; api < 4; api++) for (int f = 0; f < 3; f++) for (int state = 0; state < 4; state++) {
            g_present[0] = names[n][0]; g_present[1] = names[n][1]; g_present[2] = NULL;
            stream_proc_result = state == 1 ? CUDA_ERROR_NOT_FOUND : CUDA_SUCCESS;
            stream_proc_pointer = state != 2;
            stream_proc_status = state == 3 ? CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT : CU_GET_PROC_ADDRESS_SUCCESS;
            stream_proc_name = NULL;
            void *pfn = NULL;
            CUdriverProcAddressQueryResult status = STATUS_POISON;
            const char *symbol = names[n][explicit_pt];
            CUresult result;
            if (api == 0) result = cuGetProcAddress(symbol, &pfn, 13030, flags[f]);
            else if (api == 1) result = _cuGetProcAddress(symbol, &pfn, 13030, flags[f]);
            else if (api == 2) result = cuGetProcAddress_v2(symbol, &pfn, 13030, flags[f], &status);
            else result = _cuGetProcAddress_v2(symbol, &pfn, 13030, flags[f], &status);
            int available = stream_proc_result == CUDA_SUCCESS && stream_proc_pointer && (api < 2 || state != 3);
            int per_thread = explicit_pt || f == 2;
            void *expected = available ? (void *)names[n][per_thread] : NULL;
            if (result != stream_proc_result || pfn != expected || stream_proc_name != symbol ||
                stream_proc_flags != flags[f] || stream_proc_version != 13030 ||
                (api >= 2 && status != stream_proc_status)) {
                printf("FAIL stream lookup %s api=%d flags=%u state=%d\n", symbol, api,
                       (unsigned int)flags[f], state);
                failures++;
            }
        }
    return failures;
}

/** Run CUDA version mapping and stream lookup regressions. */
int main(void) {
    /* Boundaries from cudaTypedefs.h: cuCtxCreate _v2 / _v3 from 11040 / _v4
       from 12050, cuMemAdvise and cuMemPrefetchAsync _v2 from 12020. A CUDA
       12 build compiles in every spelling, a CUDA 13 build only the newest. */
    static const char *const ctx12[] = {"cuCtxCreate_v2", "cuCtxCreate_v3",
                                        "cuCtxCreate_v4"};
    static const char *const ctx13[] = {"cuCtxCreate_v4", NULL, NULL};
    static const char *const advise12[] = {"cuMemAdvise", "cuMemAdvise_v2", NULL};
    static const char *const advise13[] = {"cuMemAdvise_v2", NULL, NULL};
    static const char *const prefetch12[] = {"cuMemPrefetchAsync",
                                             "cuMemPrefetchAsync_v2", NULL};
    static const char *const prefetch13[] = {"cuMemPrefetchAsync_v2", NULL, NULL};

    static const Case ctx_on12[] = {
        {10000, "cuCtxCreate_v2"}, {11039, "cuCtxCreate_v2"},
        {11040, "cuCtxCreate_v3"}, {12049, "cuCtxCreate_v3"},
        {12050, "cuCtxCreate_v4"}, {12090, "cuCtxCreate_v4"},
        {13030, "cuCtxCreate_v4"}};
    static const Case ctx_on13[] = {
        {11039, NULL}, {11040, NULL}, {12049, NULL},
        {12050, "cuCtxCreate_v4"}, {12090, "cuCtxCreate_v4"},
        {13030, "cuCtxCreate_v4"}};
    static const Case advise_on12[] = {
        {10000, "cuMemAdvise"}, {12019, "cuMemAdvise"},
        {12020, "cuMemAdvise_v2"}, {12090, "cuMemAdvise_v2"},
        {13030, "cuMemAdvise_v2"}};
    static const Case advise_on13[] = {
        {12019, NULL}, {12020, "cuMemAdvise_v2"}, {13030, "cuMemAdvise_v2"}};
    static const Case prefetch_on12[] = {
        {10000, "cuMemPrefetchAsync"}, {12019, "cuMemPrefetchAsync"},
        {12020, "cuMemPrefetchAsync_v2"}, {12090, "cuMemPrefetchAsync_v2"},
        {13030, "cuMemPrefetchAsync_v2"}};
    static const Case prefetch_on13[] = {
        {12019, NULL}, {12020, "cuMemPrefetchAsync_v2"},
        {13030, "cuMemPrefetchAsync_v2"}};

    int failures = 0;
    failures += check_cases("cuCtxCreate", ctx12, "cuda12 build", ctx_on12, 7);
    failures += check_cases("cuCtxCreate", ctx13, "cuda13 build", ctx_on13, 6);
    failures += check_cases("cuMemAdvise", advise12, "cuda12 build", advise_on12, 5);
    failures += check_cases("cuMemAdvise", advise13, "cuda13 build", advise_on13, 3);
    failures += check_cases("cuMemPrefetchAsync", prefetch12, "cuda12 build",
                            prefetch_on12, 5);
    failures += check_cases("cuMemPrefetchAsync", prefetch13, "cuda13 build",
                            prefetch_on13, 3);
    failures += check_symbol_status();
    failures += check_stream_procs();

    if (failures) {
        printf("%d version map check(s) failed\n", failures);
        return 1;
    }
    printf("all version map checks passed\n");
    return 0;
}

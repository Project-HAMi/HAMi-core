// Copyright 2026 The HAMi Authors.
#include <cuda.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MIB (1ull << 20)
#define CHECK(call) do { \
    CUresult result = (call); \
    if (result != CUDA_SUCCESS) { \
        fprintf(stderr, "FAIL line %d: %s = %d\n", __LINE__, #call, result); \
        return 1; \
    } \
} while (0)

/** Remove only the private cache created by this test's launcher. */
static void remove_test_cache(void) {
    const char *cache = getenv("HAMI_GRAPH_TEST_CACHE");
    if (cache != NULL && strncmp(cache, "/tmp/hami-graph-output-", 23) == 0) unlink(cache);
}

/** Relaunch with an isolated preload and a one-GiB quota after checking GPU availability. */
static int launch_preloaded(char **argv) {
    int count = 0, pools = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) {
        puts("SKIP: no usable NVIDIA GPU");
        return 77;
    }
    CHECK(cuDeviceGetAttribute(&pools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, 0));
    if (!pools) {
        puts("SKIP: graph memory is unsupported");
        return 77;
    }
    const char *lib = getenv("LIBVGPU");
    if (lib == NULL) {
        fputs("LIBVGPU is required\n", stderr);
        return 1;
    }
    char cache[] = "/tmp/hami-graph-output-XXXXXX";
    int fd = mkstemp(cache);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    close(fd); unlink(cache);
    setenv("LD_PRELOAD", lib, 1); setenv("CUDA_REDIRECT", lib, 1);
    setenv("CUDA_DEVICE_MEMORY_LIMIT", "1024m", 1);
    setenv("CUDA_DEVICE_MEMORY_SHARED_CACHE", cache, 1); setenv("HAMI_GRAPH_TEST_CACHE", cache, 1);
    setenv("CUDA_DEVICE_SM_LIMIT", "100", 1); setenv("LIBVGPU_HOSTPID_BROKER", "0", 1);
    char *args[] = {argv[0], argv[1], "--preloaded", NULL};
    execv(argv[0], args);
    perror("execv"); unlink(cache);
    return 1;
}

/** Add a graph allocation and optionally its matching free node. */
static CUresult create_allocation(CUgraph *graph, CUdeviceptr *ptr, size_t mib, int temporary) {
    CUresult result = cuGraphCreate(graph, 0);
    if (result != CUDA_SUCCESS) return result;
    CUDA_MEM_ALLOC_NODE_PARAMS params = {0}; CUgraphNode node;
    params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE; params.poolProps.location.id = 0;
    params.bytesize = mib * MIB;
    result = cuGraphAddMemAllocNode(&node, *graph, NULL, 0, &params);
    if (result == CUDA_SUCCESS) *ptr = params.dptr;
    if (result == CUDA_SUCCESS && temporary) {
        CUgraphNode free_node;
        result = cuGraphAddMemFreeNode(&free_node, *graph, &node, 1, params.dptr);
    }
    return result;
}

/** Wait for GPU work without advancing HAMi's accounting poll. */
static CUresult native_stream_sync(CUstream stream) {
    const char *versions[] = {"GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.2.5"};
    void *(*lookup)(void *, const char *) = NULL;
    for (size_t i = 0; i < sizeof(versions) / sizeof(versions[0]) && lookup == NULL; i++)
        lookup = dlvsym(RTLD_NEXT, "dlsym", versions[i]);
    if (lookup == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (driver == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult (*sync)(CUstream) = lookup(driver, "cuStreamSynchronize");
    CUresult result = sync == NULL || sync == cuStreamSynchronize ? CUDA_ERROR_NOT_SUPPORTED : sync(stream);
    dlclose(driver);
    return result;
}

/** Check retired roots remain independent after the pending cache is rebuilt. */
static int check_retired_roots(void) {
    CUgraph graphs[2], other; CUgraphExec execs[2]; CUdeviceptr ptr, ordinary;
    for (int i = 0; i < 2; i++) {
        CHECK(create_allocation(&graphs[i], &ptr, 400, 1));
        CHECK(cuGraphInstantiateWithFlags(&execs[i], graphs[i], 0));
        CHECK(cuGraphDestroy(graphs[i]));
    }
    CHECK(create_allocation(&other, &ptr, 500, 1)); CHECK(cuGraphDestroy(other));
    CHECK(cuMemAlloc_v2(&ordinary, 300 * MIB)); CHECK(cuMemFree_v2(ordinary));
    for (int i = 0; i < 2; i++) CHECK(cuGraphExecDestroy(execs[i]));
    puts("PASS: retired roots permit a valid ordinary allocation");
    return 0;
}

/** Retain proof of a completed external free before its context destroys the event. */
static int check_context_free(CUstream origin_stream) {
    CUcontext origin, other; CUgraph a, b; CUgraphExec x, y; CUdeviceptr ptr, ordinary;
    CHECK(cuCtxGetCurrent(&origin));
    CHECK(create_allocation(&a, &ptr, 600, 0)); CHECK(cuGraphInstantiateWithFlags(&x, a, 0));
    CHECK(cuGraphLaunch(x, origin_stream)); CHECK(native_stream_sync(origin_stream));
#if CUDA_VERSION >= 13000
    CHECK(cuCtxCreate(&other, NULL, 0, 0));
#else
    CHECK(cuCtxCreate(&other, 0, 0));
#endif
    CUstream free_stream; CUgraphNode node;
    CHECK(cuStreamCreate(&free_stream, CU_STREAM_NON_BLOCKING)); CHECK(cuGraphCreate(&b, 0));
    CHECK(cuGraphAddMemFreeNode(&node, b, NULL, 0, ptr)); CHECK(cuGraphInstantiateWithFlags(&y, b, 0));
    CHECK(cuGraphLaunch(y, free_stream)); CHECK(native_stream_sync(free_stream));
    CHECK(cuCtxDestroy_v2(other)); CHECK(cuCtxSetCurrent(origin)); CHECK(cuDeviceGraphMemTrim(0));
    CHECK(cuMemAlloc_v2(&ordinary, 300 * MIB)); CHECK(cuMemFree_v2(ordinary));
    CHECK(cuGraphExecDestroy(y)); CHECK(cuGraphDestroy(b));
    CHECK(cuGraphExecDestroy(x)); CHECK(cuGraphDestroy(a));
    puts("PASS: context teardown preserves the completed external free");
    return 0;
}

/** Exercise persistent quota admission and retry against a real CUDA driver. */
int main(int argc, char **argv) {
    if (argc < 2) return 1;
    if (argc == 2) return launch_preloaded(argv);
    atexit(remove_test_cache);
    int mode = strcmp(argv[1], "launch") == 0 ? 0 : strcmp(argv[1], "upload") == 0 ? 1 : 2;
    uint64_t (*usage)(int) = dlsym(RTLD_DEFAULT, "get_current_device_memory_usage");
    uint64_t (*limit)(int) = dlsym(RTLD_DEFAULT, "get_current_device_memory_limit");
    if (usage == NULL || limit == NULL) {
        fputs("FAIL: preload is missing\n", stderr);
        return 1;
    }
    CHECK(cuInit(0));
    CUcontext ctx; CUstream stream;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, 0)); CHECK(cuCtxSetCurrent(ctx));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    if (strcmp(argv[1], "retired-root") == 0 || strcmp(argv[1], "context-free") == 0) {
        int result = strcmp(argv[1], "retired-root") == 0 ? check_retired_roots() : check_context_free(stream);
        CHECK(cuStreamDestroy_v2(stream));
        return result;
    }
    CUgraph graphs[2]; CUgraphExec execs[2] = {NULL, NULL}; CUdeviceptr ptrs[2];
    for (int i = 0; i < 2; i++) CHECK(create_allocation(&graphs[i], &ptrs[i], 512, 0));
    CUresult second;
    if (mode == 2) {
        CUDA_GRAPH_INSTANTIATE_PARAMS params = {0};
        params.flags = CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD; params.hUploadStream = stream;
        CHECK(cuGraphInstantiateWithParams(&execs[0], graphs[0], &params));
        second = cuGraphInstantiateWithParams(&execs[1], graphs[1], &params);
    } else {
        for (int i = 0; i < 2; i++) CHECK(cuGraphInstantiateWithFlags(&execs[i], graphs[i], 0));
        CUresult (*submit)(CUgraphExec, CUstream) = mode ? cuGraphUpload : cuGraphLaunch;
        CHECK(submit(execs[0], stream)); CHECK(cuStreamSynchronize(stream));
        second = submit(execs[1], stream);
    }
    CHECK(cuStreamSynchronize(stream));
    printf("mode=%s second=%d usage=%" PRIu64 " limit=%" PRIu64 "\n",
        argv[1], second, usage(0), limit(0));
    if (second != CUDA_ERROR_OUT_OF_MEMORY || usage(0) > limit(0) ||
        (mode == 2 && execs[1] != NULL)) return 1;
    if (mode != 0) {
        CHECK(cuGraphLaunch(execs[0], stream));
        CHECK(cuStreamSynchronize(stream));
    }
    CHECK(cuMemFree_v2(ptrs[0])); CHECK(cuDeviceGraphMemTrim(0));
    if (execs[1] == NULL) CHECK(cuGraphInstantiateWithFlags(&execs[1], graphs[1], 0));
    CHECK(cuGraphLaunch(execs[1], stream)); CHECK(cuMemsetD8Async(ptrs[1], 7, 4096, stream));
    CHECK(cuStreamSynchronize(stream)); CHECK(cuMemFree_v2(ptrs[1]));
    for (int i = 0; i < 2; i++) {
        CHECK(cuGraphExecDestroy(execs[i]));
        CHECK(cuGraphDestroy(graphs[i]));
    }
    CHECK(cuDeviceGraphMemTrim(0)); CHECK(cuStreamDestroy_v2(stream));
    puts("PASS: rejected second output and admitted it after the first was freed");
    return 0;
}

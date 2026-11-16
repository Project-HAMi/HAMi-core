// Copyright 2026 The HAMi Authors.
/* Call production wrappers with fake allocator and driver dependencies. */
#include <stdio.h>
#include <stdlib.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define REQUIRE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); \
        exit(1); \
    } \
} while (0)

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static CUresult native_result;
static CUstream seen_stream;
static int calls, completions, legacy_calls, checks;
static int launches, capture_starts, capture_ends;
static CUstream expected_stream;
static CUDA_GRAPH_INSTANTIATE_PARAMS *expected_params;
int pidfound = 1;
static int preparations, limits;
void pre_launch_kernel(void) { preparations++; }
/** Require the existing quota throttle for the kernel dimensions. */
void rate_limiter(int64_t grid, int64_t block) { REQUIRE(grid == 24 && block == 120); limits++; }

void ensure_initialized(void) {}
void ensure_hostpid(void) {}
int is_gpu_core_limit_enabled(void) { return 0; }
int wait_status_self(int status) { REQUIRE(status == 1); return 1; }

static CUresult record_stream(CUstream stream) {
    seen_stream = stream; calls++; return native_result;
}

int allocate_async_raw(CUdeviceptr *ptr, size_t size, CUstream stream) {
    REQUIRE(size == 32); *ptr = 101; return record_stream(stream);
}

/** Record pool-allocation arguments and the forwarded stream. */
int allocate_from_pool_async_raw(CUdeviceptr *ptr, size_t size, CUmemoryPool pool, CUstream stream) {
    REQUIRE(size == 32 && pool == (CUmemoryPool)7); *ptr = 101; return record_stream(stream);
}

int free_raw_async(CUdeviceptr ptr, CUstream stream) {
    REQUIRE(ptr == 101); return record_stream(stream);
}

/** Record cleanup after a successful asynchronous free. */
void graph_ptr_freed_async(CUdeviceptr ptr, CUstream stream) {
    REQUIRE(ptr == 101 && stream == seen_stream); completions++;
}

/** Record graph tracking after successful native instantiation. */
int graph_exec_created(CUgraph graph, CUgraphExec exec) {
    REQUIRE(graph == (CUgraph)8 && exec == (CUgraphExec)9); completions++; return 0;
}

void graph_exec_launched(CUgraphExec exec) {
    REQUIRE(exec == (CUgraphExec)9); launches++;
}

int graph_capture_prepare(CUstream stream, CUstreamCaptureMode mode, uint64_t *token) {
    (void)mode; REQUIRE(stream == expected_stream); *token = 123; return 0;
}
void graph_capture_started(uint64_t token, CUresult result) {
    REQUIRE(token == 123); capture_starts += result == CUDA_SUCCESS;
}
uint64_t graph_capture_token(CUstream stream, cuuint64_t *id) {
    REQUIRE(stream == expected_stream); *id = 123; return 123;
}
CUresult graph_capture_resolve(CUstream stream, cuuint64_t *id, uint64_t *token) {
    *token = graph_capture_token(stream, id); return CUDA_SUCCESS;
}
void graph_capture_finish(uint64_t token) { REQUIRE(token == 123); }
CUresult graph_capture_info(CUstream stream, CUstreamCaptureStatus *cap, cuuint64_t *id, CUgraph *graph) {
    (void)stream; *cap = CU_STREAM_CAPTURE_STATUS_NONE; *id = 0; *graph = NULL; return CUDA_SUCCESS;
}

/** Complete capture tracking with the native result. */
void graph_capture_end(cuuint64_t id, CUgraph graph) {
    REQUIRE(id == 123 && graph == (native_result == CUDA_SUCCESS ? (CUgraph)8 : NULL));
    capture_ends++;
}

/** Record copy arguments and the forwarded stream. */
static CUresult native_copy(CUdeviceptr dst, CUdeviceptr src, size_t size, CUstream stream) {
    REQUIRE(dst == 101 && src == 202 && size == 32); return record_stream(stream);
}

static CUresult native_submit(CUgraphExec exec, CUstream stream) {
    REQUIRE(exec == (CUgraphExec)9); return record_stream(stream);
}

static CUresult native_begin(CUstream stream, CUstreamCaptureMode mode) {
    REQUIRE(mode == CU_STREAM_CAPTURE_MODE_THREAD_LOCAL); return record_stream(stream);
}

static CUresult native_end(CUstream stream, CUgraph *graph) {
    *graph = (CUgraph)8; return record_stream(stream);
}

/** Record the per-thread instantiation parameters. */
static CUresult native_params(CUgraphExec *exec, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS *params) {
    REQUIRE(graph == (CUgraph)8 && params == expected_params);
    REQUIRE(params->flags == CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD);
    *exec = (CUgraphExec)9;
    return record_stream(params->hUploadStream);
}

static CUresult legacy_params(CUgraphExec *exec, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS *params) {
    (void)exec; (void)graph; (void)params; legacy_calls++; return native_result;
}


/** Check the host-copy arguments delivered to native CUDA. */
static CUresult native_htod(CUdeviceptr dst, const void *src, size_t size, CUstream stream) {
    REQUIRE(dst == 101 && src == (const void *)202 && size == 32); return record_stream(stream);
}

/** Check the host-readback arguments delivered to native CUDA. */
static CUresult native_dtoh(void *dst, CUdeviceptr src, size_t size, CUstream stream) {
    REQUIRE(dst == (void *)101 && src == 202 && size == 32); return record_stream(stream);
}

/** Check the memset arguments delivered to native CUDA. */
static CUresult native_set8(CUdeviceptr ptr, unsigned char value, size_t count, CUstream stream) {
    REQUIRE(ptr == 101 && value == 7 && count == 32); return record_stream(stream);
}

/** Check the pitched memset arguments delivered to native CUDA. */
static CUresult native_set2d8(CUdeviceptr ptr, size_t pitch, unsigned char value,
                                  size_t width, size_t height, CUstream stream) {
    REQUIRE(ptr == 101 && pitch == 128 && value == 7 && width == 16 && height == 2);
    return record_stream(stream);
}

/** Check the memset arguments delivered to native CUDA. */
static CUresult native_set16(CUdeviceptr ptr, uint16_t value, size_t count, CUstream stream) {
    REQUIRE(ptr == 101 && value == 7 && count == 32); return record_stream(stream);
}

/** Check the pitched memset arguments delivered to native CUDA. */
static CUresult native_set2d16(CUdeviceptr ptr, size_t pitch, uint16_t value,
                                  size_t width, size_t height, CUstream stream) {
    REQUIRE(ptr == 101 && pitch == 128 && value == 7 && width == 16 && height == 2);
    return record_stream(stream);
}

/** Check the memset arguments delivered to native CUDA. */
static CUresult native_set32(CUdeviceptr ptr, unsigned int value, size_t count, CUstream stream) {
    REQUIRE(ptr == 101 && value == 7 && count == 32); return record_stream(stream);
}

/** Check the pitched memset arguments delivered to native CUDA. */
static CUresult native_set2d32(CUdeviceptr ptr, size_t pitch, unsigned int value,
                                  size_t width, size_t height, CUstream stream) {
    REQUIRE(ptr == 101 && pitch == 128 && value == 7 && width == 16 && height == 2);
    return record_stream(stream);
}

/** Check kernel arguments and require the existing compute-limit path. */
static CUresult native_kernel(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
                               unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                               CUstream stream, void **params, void **extra) {
    REQUIRE(function == (CUfunction)8 && gx == 2 && gy == 3 && gz == 4);
    REQUIRE(bx == 4 && by == 5 && bz == 6 && shared == 32 && params == NULL && extra == NULL);
    REQUIRE(preparations == 1 && limits == 1);
    return record_stream(stream);
}

/** Exercise added copy, memset, and kernel aliases with every stream and native result. */
static void check_submission_wrappers(CUstream stream) {
    CUstream expected = stream == NULL ? CU_STREAM_PER_THREAD : stream;
    for (int op = 0; op < 10; op++) {
        calls = preparations = limits = 0;
        CUresult result = CUDA_ERROR_UNKNOWN;
        switch (op) {
            case 0: result = cuMemcpyHtoDAsync_v2_ptsz(101, (const void *)202, 32, stream); break;
            case 1: result = cuMemcpyDtoHAsync_v2_ptsz((void *)101, 202, 32, stream); break;
            case 2: result = cuMemcpyDtoDAsync_v2_ptsz(101, 202, 32, stream); break;
            case 3: result = cuMemsetD8Async_ptsz(101, 7, 32, stream); break;
            case 4: result = cuMemsetD16Async_ptsz(101, 7, 32, stream); break;
            case 5: result = cuMemsetD32Async_ptsz(101, 7, 32, stream); break;
            case 6: result = cuMemsetD2D8Async_ptsz(101, 128, 7, 16, 2, stream); break;
            case 7: result = cuMemsetD2D16Async_ptsz(101, 128, 7, 16, 2, stream); break;
            case 8: result = cuMemsetD2D32Async_ptsz(101, 128, 7, 16, 2, stream); break;
            case 9: result = cuLaunchKernel_ptsz((CUfunction)8, 2, 3, 4, 4, 5, 6, 32, stream, NULL, NULL); break;
        }
        REQUIRE(result == native_result && calls == 1 && seen_stream == expected);
        checks++;
    }
}

/** Invoke each actual wrapper and check the stream delivered to its dependencies. */
static void check_wrappers(CUstream stream) {
    CUstream expected = stream == NULL ? CU_STREAM_PER_THREAD : stream;
    expected_stream = expected;
    for (int op = 0; op < 9; op++) {
        CUdeviceptr ptr = 0;
        CUgraph graph = NULL;
        CUresult result = CUDA_ERROR_UNKNOWN;
        calls = completions = 0;
        launches = capture_starts = capture_ends = 0;
        seen_stream = (CUstream)99;
        switch (op) {
            case 0: result = cuMemAllocAsync_ptsz(&ptr, 32, stream); break;
            case 1: result = cuMemAllocFromPoolAsync_ptsz(&ptr, 32, (CUmemoryPool)7, stream); break;
            case 2: result = cuMemFreeAsync_ptsz(101, stream); break;
            case 3: result = cuMemcpyAsync_ptsz(101, 202, 32, stream); break;
            case 4: result = cuStreamSynchronize_ptsz(stream); break;
            case 5: result = cuGraphLaunch_ptsz((CUgraphExec)9, stream); break;
            case 6: result = cuGraphUpload_ptsz((CUgraphExec)9, stream); break;
            case 7: result = cuStreamBeginCapture_v2_ptsz(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL); break;
            case 8: result = cuStreamEndCapture_ptsz(stream, &graph); break;
        }
        REQUIRE(result == native_result && calls == 1 && seen_stream == expected);
        REQUIRE(completions == (op == 2 && result == CUDA_SUCCESS));
        REQUIRE(launches == ((op == 5 || op == 6) && result == CUDA_SUCCESS));
        REQUIRE(capture_starts == (op == 7 && result == CUDA_SUCCESS));
        REQUIRE(capture_ends == (op == 8));
        if ((op == 0 || op == 1) && result == CUDA_SUCCESS) REQUIRE(ptr == 101);
        if (op == 8 && result == CUDA_SUCCESS) REQUIRE(graph == (CUgraph)8);
        checks++;
    }
}

/** Ensure parameter-based instantiation uses the native per-thread entry point. */
static void check_instantiation(CUstream stream) {
    CUDA_GRAPH_INSTANTIATE_PARAMS params = {0};
    params.flags = CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD;
    params.hUploadStream = stream;
    expected_params = &params;
    CUgraphExec exec;
    calls = completions = legacy_calls = launches = 0;
    CUresult result = cuGraphInstantiateWithParams_ptsz(&exec, (CUgraph)8, &params);
    REQUIRE(result == native_result && calls == 1 && legacy_calls == 0);
    REQUIRE(seen_stream == stream && params.hUploadStream == stream);
    REQUIRE(completions == (result == CUDA_SUCCESS));
    REQUIRE(launches == (result == CUDA_SUCCESS));
    if (result == CUDA_SUCCESS) REQUIRE(exec == (CUgraphExec)9);
    checks++;
}

#include "test/extended_stream_fixture.h"

/** Run stream and result checks through all production per-thread wrappers. */
int main(void) {
    log_utils_init();
    register_extended_streams();
    cuda_library_entry[OVERRIDE_cuMemcpyAsync].fn_ptr = native_copy;
    cuda_library_entry[OVERRIDE_cuMemcpyHtoDAsync_v2].fn_ptr = native_htod;
    cuda_library_entry[OVERRIDE_cuMemcpyDtoHAsync_v2].fn_ptr = native_dtoh;
    cuda_library_entry[OVERRIDE_cuMemcpyDtoDAsync_v2].fn_ptr = native_copy;
    cuda_library_entry[OVERRIDE_cuLaunchKernel].fn_ptr = native_kernel;
    cuda_library_entry[OVERRIDE_cuMemsetD8Async].fn_ptr = native_set8;
    cuda_library_entry[OVERRIDE_cuMemsetD2D8Async].fn_ptr = native_set2d8;
    cuda_library_entry[OVERRIDE_cuMemsetD16Async].fn_ptr = native_set16;
    cuda_library_entry[OVERRIDE_cuMemsetD2D16Async].fn_ptr = native_set2d16;
    cuda_library_entry[OVERRIDE_cuMemsetD32Async].fn_ptr = native_set32;
    cuda_library_entry[OVERRIDE_cuMemsetD2D32Async].fn_ptr = native_set2d32;
    cuda_library_entry[OVERRIDE_cuStreamSynchronize].fn_ptr = record_stream;
    cuda_library_entry[OVERRIDE_cuGraphLaunch].fn_ptr = native_submit;
    cuda_library_entry[OVERRIDE_cuGraphUpload].fn_ptr = native_submit;
    cuda_library_entry[OVERRIDE_cuStreamBeginCapture_v2].fn_ptr = native_begin;
    cuda_library_entry[OVERRIDE_cuStreamEndCapture].fn_ptr = native_end;
    cuda_library_entry[OVERRIDE_cuGraphInstantiateWithParams].fn_ptr = legacy_params;
    cuda_library_entry[OVERRIDE_cuGraphInstantiateWithParams_ptsz].fn_ptr = native_params;
    const CUstream streams[] = {NULL, (CUstream)77, CU_STREAM_LEGACY, CU_STREAM_PER_THREAD};
    for (int error = 0; error < 2; error++) {
        native_result = error ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
        for (size_t i = 0; i < sizeof(streams) / sizeof(streams[0]); i++) {
            check_wrappers(streams[i]);
            check_instantiation(streams[i]);
            check_submission_wrappers(streams[i]);
            check_extended_streams(streams[i]);
        }
    }
    printf("PASS: %d production wrapper calls preserve stream semantics and results\n", checks);
    return 0;
}

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

void graph_capture_begin(CUstream stream) {
    REQUIRE(stream == expected_stream); capture_starts++;
}

/** Return the capture associated with the normalized stream. */
cuuint64_t graph_capture_id(CUstream stream, int take) {
    REQUIRE(stream == expected_stream && take == 1); return 123;
}

/** Complete capture tracking with the native result. */
void graph_capture_end(cuuint64_t id, CUgraph graph) {
    REQUIRE(id == 123 && graph == (native_result == CUDA_SUCCESS ? (CUgraph)8 : NULL));
    capture_ends++;
}

typedef struct graph_exec_state graph_exec_state;

int graph_exec_prepare(CUgraph graph, graph_exec_state **state) {
    REQUIRE(graph == (CUgraph)8); *state = (graph_exec_state *)4; return 0;
}
int graph_exec_reserve(graph_exec_state *state, CUstream stream) {
    (void)state; REQUIRE(stream == expected_stream); return 0;
}
void graph_exec_commit(graph_exec_state *state, CUgraphExec exec) {
    (void)state; REQUIRE(exec == (CUgraphExec)9); completions++;
}
void graph_exec_discard(graph_exec_state *state) { (void)state; }
int graph_exec_begin(CUgraphExec exec, CUstream stream, graph_exec_state **state) {
    REQUIRE(exec == (CUgraphExec)9); REQUIRE(stream == expected_stream);
    *state = (graph_exec_state *)5; return 0;
}
void graph_exec_end(graph_exec_state *state, CUstream stream, CUresult result) {
    (void)state; REQUIRE(stream == expected_stream); launches += result == CUDA_SUCCESS;
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
void graph_capture_finish(uint64_t token) { REQUIRE(token == 123); }
CUresult graph_capture_info(CUstream stream, CUstreamCaptureStatus *cap, cuuint64_t *id, CUgraph *graph) {
    (void)stream; *cap = CU_STREAM_CAPTURE_STATUS_NONE; *id = 0; *graph = NULL; return CUDA_SUCCESS;
}
uint64_t graph_sync_begin(void) { return 1; }
void graph_synchronized(CUcontext ctx, CUstream stream, int whole, uint64_t serial) {
    (void)ctx; (void)whole; (void)serial; REQUIRE(stream == seen_stream);
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
        REQUIRE(completions == 0);
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

/** Run stream and result checks through all production per-thread wrappers. */
int main(void) {
    log_utils_init();
    cuda_library_entry[OVERRIDE_cuMemcpyAsync].fn_ptr = native_copy;
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
        }
    }
    printf("PASS: %d production wrapper calls preserve stream semantics and results\n", checks);
    return 0;
}

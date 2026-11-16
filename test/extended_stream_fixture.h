// Copyright 2026 The HAMi Authors.
static const void *expected_copy;
static int sync_calls;

/** Verify an extended launch keeps dimensions, attributes, and compute limiting. */
static CUresult native_kernel_ex(const CUlaunchConfig *config, CUfunction f, void **params, void **extra) {
    REQUIRE(config != NULL && f == (CUfunction)8 && params == NULL && extra == NULL);
    REQUIRE(config->gridDimX == 2 && config->gridDimY == 3 && config->gridDimZ == 4);
    REQUIRE(config->blockDimX == 4 && config->blockDimY == 5 && config->blockDimZ == 6);
    REQUIRE(config->sharedMemBytes == 32 && config->attrs == (CUlaunchAttribute *)3
         && config->numAttrs == 1);  // NOLINT(readability/casting)
    REQUIRE(preparations == 1 && limits == 1);
    return record_stream(config->hStream);
}

static CUresult native_cooperative(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                                   unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                                   CUstream stream, void **params) {
    return native_kernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, NULL);
}

/** Verify copy descriptors and native completion results without translation. */
static CUresult native_copy2d(const CUDA_MEMCPY2D *copy, CUstream stream) {
    REQUIRE(copy == expected_copy); return record_stream(stream);
}
static CUresult native_copy3d(const CUDA_MEMCPY3D *copy, CUstream stream) {
    REQUIRE(copy == expected_copy); return record_stream(stream);
}
static CUresult native_copy3d_peer(const CUDA_MEMCPY3D_PEER *copy, CUstream stream) {
    REQUIRE(copy == expected_copy); return record_stream(stream);
}

/** Preserve peer device contexts in asynchronous copies. */
static CUresult native_peer(CUdeviceptr dst, CUcontext dst_ctx, CUdeviceptr src, CUcontext src_ctx,
                            size_t size, CUstream stream) {
    REQUIRE(dst_ctx == (CUcontext)3 && src_ctx == (CUcontext)4);
    return native_copy(dst, src, size, stream);
}

/** Validate the original prefetch ABI without reinterpreting the destination. */
static CUresult native_prefetch(CUdeviceptr ptr, size_t count, CUdevice dev, CUstream stream) {
    REQUIRE(ptr == 101 && count == 32 && dev == 7); return record_stream(stream);
}

/** Validate the location-based prefetch ABI without changing its flags. */
static CUresult native_prefetch_v2(CUdeviceptr ptr, size_t count, CUmemLocation location,
                                   unsigned int flags, CUstream stream) {
    REQUIRE(location.type == CU_MEM_LOCATION_TYPE_DEVICE && location.id == 7 && flags == 0);
    return native_prefetch(ptr, count, location.id, stream);
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t size) {
    REQUIRE(dst == 101 && src == 202 && size == 32); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t size) {
    REQUIRE(dst == 101 && src == (const void *)202  // NOLINT(readability/casting)
         && size == 32); sync_calls++; return native_result;  // NOLINT(readability/casting)
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t size) {
    REQUIRE(dst == (void *)101 && src == 202 && size == 32);  // NOLINT(readability/casting)
    sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t size) {
    REQUIRE(dst == 101 && src == 202 && size == 32); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpyPeer(CUdeviceptr dst, CUcontext dst_ctx, CUdeviceptr src, CUcontext src_ctx, size_t size) {
    REQUIRE(dst == 101 && src == 202 && size == 32 && dst_ctx == (CUcontext)3
         && src_ctx == (CUcontext)4); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpy2D_v2(const CUDA_MEMCPY2D *copy) {
    REQUIRE(copy == expected_copy); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *copy) {
    REQUIRE(copy == expected_copy); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpy3D_v2(const CUDA_MEMCPY3D *copy) {
    REQUIRE(copy == expected_copy); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *copy) {
    REQUIRE(copy == expected_copy); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemsetD8_v2(CUdeviceptr ptr, unsigned char value, size_t count) {
    REQUIRE(ptr == 101 && value == 7 && count == 32); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemsetD2D8_v2(CUdeviceptr ptr, size_t pitch, unsigned char value, size_t width, size_t height) {
    REQUIRE(ptr == 101 && pitch == 128 && value == 7 && width == 16 && height == 2); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemsetD16_v2(CUdeviceptr ptr, uint16_t value, size_t count) {
    REQUIRE(ptr == 101 && value == 7 && count == 32); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemsetD2D16_v2(CUdeviceptr ptr, size_t pitch, uint16_t value, size_t width, size_t height) {
    REQUIRE(ptr == 101 && pitch == 128 && value == 7 && width == 16 && height == 2); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemsetD32_v2(CUdeviceptr ptr, unsigned int value, size_t count) {
    REQUIRE(ptr == 101 && value == 7 && count == 32); sync_calls++; return native_result;
}

/** Verify the exact native synchronous per-thread entry is called. */
static CUresult sync_cuMemsetD2D32_v2(CUdeviceptr ptr, size_t pitch, unsigned int value, size_t width, size_t height) {
    REQUIRE(ptr == 101 && pitch == 128 && value == 7 && width == 16 && height == 2); sync_calls++; return native_result;
}

/** Exercise the remaining stream aliases across streams, ABIs, and native errors. */
static void check_extended_streams(CUstream stream) {
    CUlaunchConfig config = {0};
    config.gridDimX = 2; config.gridDimY = 3; config.gridDimZ = 4;
    config.blockDimX = 4; config.blockDimY = 5; config.blockDimZ = 6;
    config.sharedMemBytes = 32; config.hStream = stream;
    config.attrs = (CUlaunchAttribute *)3; config.numAttrs = 1;  // NOLINT(readability/casting)
    CUDA_MEMCPY2D copy = {0}; expected_copy = &copy;
    CUmemLocation location = {CU_MEM_LOCATION_TYPE_DEVICE, 7};
    for (int op = 0; op < 8; op++) {
        calls = preparations = limits = 0;
        CUresult result = CUDA_ERROR_UNKNOWN;
        switch (op) {
            case 0: result = cuLaunchKernelEx_ptsz(&config, (CUfunction)8, NULL, NULL); break;
            case 1: result = cuLaunchCooperativeKernel_ptsz((CUfunction)8, 2, 3, 4, 4, 5, 6,
                32, stream, NULL); break;
            case 2: result = cuMemcpy2DAsync_v2_ptsz(&copy, stream); break;
            case 3: result = cuMemcpy3DAsync_v2_ptsz((const CUDA_MEMCPY3D *)&copy, stream); break;
            case 4: result = cuMemcpy3DPeerAsync_ptsz((const CUDA_MEMCPY3D_PEER *)&copy, stream); break;
            case 5: result = cuMemPrefetchAsync_ptsz(101, 32, 7, stream); break;
            case 6: result = cuMemPrefetchAsync_v2_ptsz(101, 32, location, 0, stream); break;
            case 7: result = cuMemcpyPeerAsync_ptsz(101, (CUcontext)3, 202, (CUcontext)4, 32, stream); break;
        }
        REQUIRE(result == native_result && calls == 1);
        REQUIRE(seen_stream == (stream == NULL ? CU_STREAM_PER_THREAD : stream));
        REQUIRE(config.hStream == stream); checks++;
    }
    sync_calls = 0;
    REQUIRE(cuMemcpy_ptds(101, 202, 32) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemcpyHtoD_v2_ptds(101, (const void *)202, 32) == native_result  // NOLINT(readability/casting)
         && sync_calls == 1); checks++;  // NOLINT(readability/casting)
    sync_calls = 0;
    void *host = (void *)101;  // NOLINT(readability/casting)
    REQUIRE(cuMemcpyDtoH_v2_ptds(host, 202, 32) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemcpyDtoD_v2_ptds(101, 202, 32) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemcpyPeer_ptds(101, (CUcontext)3, 202, (CUcontext)4, 32) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemcpy2D_v2_ptds((const CUDA_MEMCPY2D *)expected_copy) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemcpy2DUnaligned_v2_ptds((const CUDA_MEMCPY2D *)expected_copy) == native_result
         && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemcpy3D_v2_ptds((const CUDA_MEMCPY3D *)expected_copy) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemcpy3DPeer_ptds((const CUDA_MEMCPY3D_PEER *)expected_copy) == native_result
         && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemsetD8_v2_ptds(101, 7, 32) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemsetD2D8_v2_ptds(101, 128, 7, 16, 2) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemsetD16_v2_ptds(101, 7, 32) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemsetD2D16_v2_ptds(101, 128, 7, 16, 2) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemsetD32_v2_ptds(101, 7, 32) == native_result && sync_calls == 1); checks++;
    sync_calls = 0;
    REQUIRE(cuMemsetD2D32_v2_ptds(101, 128, 7, 16, 2) == native_result && sync_calls == 1); checks++;
}

static void register_extended_streams(void) {
    cuda_library_entry[OVERRIDE_cuLaunchKernelEx].fn_ptr = native_kernel_ex;
    cuda_library_entry[OVERRIDE_cuLaunchCooperativeKernel].fn_ptr = native_cooperative;
    cuda_library_entry[OVERRIDE_cuMemcpy2DAsync_v2].fn_ptr = native_copy2d;
    cuda_library_entry[OVERRIDE_cuMemcpy3DAsync_v2].fn_ptr = native_copy3d;
    cuda_library_entry[OVERRIDE_cuMemcpy3DPeerAsync].fn_ptr = native_copy3d_peer;
    cuda_library_entry[OVERRIDE_cuMemcpyPeerAsync].fn_ptr = native_peer;
    cuda_library_entry[OVERRIDE_cuMemPrefetchAsync].fn_ptr = native_prefetch;
    cuda_library_entry[OVERRIDE_cuMemPrefetchAsync_ptsz].fn_ptr = native_prefetch;
    cuda_library_entry[OVERRIDE_cuMemPrefetchAsync_v2].fn_ptr = native_prefetch_v2;
    cuda_library_entry[OVERRIDE_cuMemcpy_ptds].fn_ptr = sync_cuMemcpy;
    cuda_library_entry[OVERRIDE_cuMemcpyHtoD_v2_ptds].fn_ptr = sync_cuMemcpyHtoD_v2;
    cuda_library_entry[OVERRIDE_cuMemcpyDtoH_v2_ptds].fn_ptr = sync_cuMemcpyDtoH_v2;
    cuda_library_entry[OVERRIDE_cuMemcpyDtoD_v2_ptds].fn_ptr = sync_cuMemcpyDtoD_v2;
    cuda_library_entry[OVERRIDE_cuMemcpyPeer_ptds].fn_ptr = sync_cuMemcpyPeer;
    cuda_library_entry[OVERRIDE_cuMemcpy2D_v2_ptds].fn_ptr = sync_cuMemcpy2D_v2;
    cuda_library_entry[OVERRIDE_cuMemcpy2DUnaligned_v2_ptds].fn_ptr = sync_cuMemcpy2DUnaligned_v2;
    cuda_library_entry[OVERRIDE_cuMemcpy3D_v2_ptds].fn_ptr = sync_cuMemcpy3D_v2;
    cuda_library_entry[OVERRIDE_cuMemcpy3DPeer_ptds].fn_ptr = sync_cuMemcpy3DPeer;
    cuda_library_entry[OVERRIDE_cuMemsetD8_v2_ptds].fn_ptr = sync_cuMemsetD8_v2;
    cuda_library_entry[OVERRIDE_cuMemsetD2D8_v2_ptds].fn_ptr = sync_cuMemsetD2D8_v2;
    cuda_library_entry[OVERRIDE_cuMemsetD16_v2_ptds].fn_ptr = sync_cuMemsetD16_v2;
    cuda_library_entry[OVERRIDE_cuMemsetD2D16_v2_ptds].fn_ptr = sync_cuMemsetD2D16_v2;
    cuda_library_entry[OVERRIDE_cuMemsetD32_v2_ptds].fn_ptr = sync_cuMemsetD32_v2;
    cuda_library_entry[OVERRIDE_cuMemsetD2D32_v2_ptds].fn_ptr = sync_cuMemsetD2D32_v2;
}

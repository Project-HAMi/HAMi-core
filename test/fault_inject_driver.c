// A fault-injecting stand-in for libcuda.so.1, used only by
// test_alloc_async_failure_leak to make the allocator's *internal*
// post-allocation driver calls fail on demand.
//
// Why this exists: account_async_chunk()/add_chunk_async() in
// src/allocator/allocator.c call cuDeviceGetMemPool()/cuMemPoolGetAttribute()
// through CUDA_OVERRIDE_CALL(cuda_library_entry, ...) -- i.e. directly
// against the function-pointer table src/cuda/hook.c's load_cuda_libraries()
// fills in by dlopen("libcuda.so.1") + dlsym(). That table lookup happens
// *inside* libvgpu.so and is never routed back through LD_PRELOAD, so an
// LD_PRELOAD shim placed "above" libvgpu.so cannot reach it. The only seam
// available is the dlopen("libcuda.so.1") call itself: if a directory
// containing a library built from this file and literally named
// "libcuda.so.1" is placed ahead of the real driver's directory in
// LD_LIBRARY_PATH, that dlopen() loads *this* file instead.
//
// This file only defines cuDeviceGetMemPool and cuMemPoolGetAttribute. Every
// other symbol load_cuda_libraries() looks up (cuMemAllocAsync,
// cuMemFreeAsync, cuInit, cuMemGetInfo_v2, ...) is absent here, so
// dlsym(table, name) returns NULL and load_cuda_libraries() falls back to
// dlsym(RTLD_NEXT, name) -- which finds the real driver linked into the test
// executable via -lcuda. So every driver call except the two below still
// talks to real hardware.
//
// Fault injection is armed with plain environment variables rather than a
// shared dlopen handle, since env vars are visible to this library
// regardless of how many times/where it ends up mapped in the process:
//   HAMI_TEST_FAIL_NEXT_GETMEMPOOL=1  -> the next cuDeviceGetMemPool call
//                                        fails once, then clears itself
//   HAMI_TEST_FAIL_NEXT_POOLATTR=1    -> the next cuMemPoolGetAttribute call
//                                        fails once, then clears itself
// The test arms one of these with setenv() immediately before calling the
// hooked cuMemAllocAsync/cuMemAllocFromPoolAsync entry point.

#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdlib.h>

typedef CUresult (*cuDeviceGetMemPool_t)(CUmemoryPool *, CUdevice);
typedef CUresult (*cuMemPoolGetAttribute_t)(CUmemoryPool, CUmemPool_attribute, void *);

/* Absolute paths real NVIDIA driver installs place libcuda.so.1 at, tried in
 * order. HAMI_TEST_REAL_LIBCUDA overrides all of them when set, for setups
 * that install the driver somewhere else. */
static const char *const kRealDriverCandidates[] = {
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/lib/wsl/lib/libcuda.so.1",
    "/usr/local/cuda/compat/libcuda.so.1",
    NULL,
};

static void *real_driver_handle(void) {
    static void *handle = NULL;
    static int tried = 0;
    if (tried)
        return handle;
    tried = 1;

    const char *override = getenv("HAMI_TEST_REAL_LIBCUDA");
    if (override && *override) {
        handle = dlopen(override, RTLD_NOW | RTLD_NODELETE);
        if (handle)
            return handle;
    }
    for (int i = 0; kRealDriverCandidates[i] != NULL; i++) {
        handle = dlopen(kRealDriverCandidates[i], RTLD_NOW | RTLD_NODELETE);
        if (handle)
            return handle;
    }
    return handle;
}

static void *real_symbol(const char *name) {
    void *h = real_driver_handle();
    return h ? dlsym(h, name) : NULL;
}

CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
    if (getenv("HAMI_TEST_FAIL_NEXT_GETMEMPOOL")) {
        unsetenv("HAMI_TEST_FAIL_NEXT_GETMEMPOOL");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    cuDeviceGetMemPool_t real = (cuDeviceGetMemPool_t)real_symbol("cuDeviceGetMemPool");
    if (!real)
        return CUDA_ERROR_UNKNOWN;
    return real(pool, dev);
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    if (getenv("HAMI_TEST_FAIL_NEXT_POOLATTR")) {
        unsetenv("HAMI_TEST_FAIL_NEXT_POOLATTR");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    cuMemPoolGetAttribute_t real = (cuMemPoolGetAttribute_t)real_symbol("cuMemPoolGetAttribute");
    if (!real)
        return CUDA_ERROR_UNKNOWN;
    return real(pool, attr, value);
}

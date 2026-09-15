// A fault-injecting companion for the real CUDA driver, used only by
// test_alloc_async_failure_leak to make the allocator's *internal*
// post-allocation driver calls fail on demand.
//
// Why this exists: account_async_chunk()/add_chunk_async() in
// src/allocator/allocator.c call cuDeviceGetMemPool()/cuMemPoolGetAttribute()
// through CUDA_OVERRIDE_CALL(cuda_library_entry, ...) -- i.e. via a function
// pointer that src/cuda/hook.c's load_cuda_libraries() resolves once, up
// front, with:
//     table = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NODELETE);
//     cuda_library_entry[i].fn_ptr = real_dlsym(table, name);
// dlsym() against an explicit handle only ever finds symbols that specific
// object exports -- it is not affected by LD_PRELOAD. So the only seam
// available to fault-inject these two calls is the dlopen("libcuda.so.1")
// call itself.
//
// An earlier version of this file tried to exploit that by being findable
// AS "libcuda.so.1" (via LD_LIBRARY_PATH ordering). That backfired: once
// something resolves to this file under that name, EVERY lookup of
// "libcuda.so.1" in the process -- including the test executable's own
// -lcuda link, resolved by the loader before main() even runs -- collapses
// onto this incomplete stand-in, and the real driver never gets loaded into
// the process at all. Since load_cuda_libraries()'s fallback for symbols
// this file doesn't define is dlsym(RTLD_NEXT, name), and RTLD_NEXT has
// nothing to find without a genuinely-loaded real driver, every other
// CUDA_OVERRIDE_CALL entry could end up NULL and the test could fail before
// either injected failure path even runs.
//
// This version instead interposes dlopen() itself (LD_PRELOAD symbol
// interposition -- the same technique src/libvgpu.c already uses for its own
// dlsym() hook). The real driver is always opened and left loaded normally,
// with LD_LIBRARY_PATH untouched, exactly as for every other test; only the
// one dlopen("libcuda.so.1", ...) call made by load_cuda_libraries() gets
// handed a handle to *this* shared object instead of the real driver's. That
// object exports exactly two CUDA symbols -- cuDeviceGetMemPool and
// cuMemPoolGetAttribute -- so dlsym(table, name) finds those two here and
// returns NULL for every other one of the ~200 entries in cuda_library_entry,
// which is exactly the case load_cuda_libraries() already handles by falling
// back to dlsym(RTLD_NEXT, name) -- and since the real driver is genuinely
// loaded elsewhere in the process, that fallback finds it correctly.
//
// Load order: LD_PRELOAD this shim ahead of libvgpu.so (prepended to the
// existing LD_PRELOAD list, not replacing it). This must not replace
// libvgpu.so's own LD_PRELOAD entry -- the test still needs cuMemAllocAsync/
// cuMemAllocFromPoolAsync to go through allocator.c's real hooks.
//
// Fault injection is armed with plain environment variables rather than a
// shared dlopen handle, since env vars are visible to this library no matter
// how many times/where it ends up mapped in the process:
//   HAMI_TEST_FAIL_NEXT_GETMEMPOOL=1  -> the next cuDeviceGetMemPool call
//                                        fails once, then clears itself
//   HAMI_TEST_FAIL_NEXT_POOLATTR=1    -> the next cuMemPoolGetAttribute call
//                                        fails once, then clears itself
// The test arms one of these with setenv() immediately before calling the
// hooked cuMemAllocAsync/cuMemAllocFromPoolAsync entry point.

#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VISIBLE __attribute__((visibility("default")))

typedef void *(*dlopen_t)(const char *, int);
typedef CUresult (*cuDeviceGetMemPool_t)(CUmemoryPool *, CUdevice);
typedef CUresult (*cuMemPoolGetAttribute_t)(CUmemoryPool, CUmemPool_attribute, void *);

static dlopen_t real_dlopen(void) {
    static dlopen_t fn = NULL;
    if (!fn)
        fn = (dlopen_t)dlsym(RTLD_NEXT, "dlopen");
    return fn;
}

/* A handle to this shared object itself -- so that dlsym() against it can
 * only ever find the two symbols defined below, nothing else. */
static void *self_handle(void) {
    static void *h = NULL;
    static int tried = 0;
    if (tried)
        return h;
    tried = 1;
    Dl_info info;
    if (dladdr((void *)self_handle, &info) && info.dli_fname)
        h = real_dlopen()(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    return h;
}

/* The genuine real-driver handle, captured the moment
 * load_cuda_libraries() asks to open it. Reused below so
 * cuDeviceGetMemPool/cuMemPoolGetAttribute can forward to the real
 * implementation without any further name-based lookup or ordering
 * assumptions relative to libvgpu.so. */
static void *g_real_driver_handle = NULL;

static void *real_driver_handle(void) {
    if (!g_real_driver_handle)
        g_real_driver_handle = real_dlopen()("libcuda.so.1", RTLD_NOW | RTLD_NODELETE);
    return g_real_driver_handle;
}

VISIBLE void *dlopen(const char *filename, int flags) {
    void *real_result = real_dlopen()(filename, flags);
    if (real_result && filename && strcmp(filename, "libcuda.so.1") == 0) {
        g_real_driver_handle = real_result;
        void *self = self_handle();
        if (self) {
            fprintf(stderr,
                    "[fault_inject_driver] substituting for libcuda.so.1: "
                    "only cuDeviceGetMemPool/cuMemPoolGetAttribute are "
                    "faked, everything else falls back to the real driver\n");
            return self;
        }
    }
    return real_result;
}

VISIBLE CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
    if (getenv("HAMI_TEST_FAIL_NEXT_GETMEMPOOL")) {
        unsetenv("HAMI_TEST_FAIL_NEXT_GETMEMPOOL");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    void *h = real_driver_handle();
    cuDeviceGetMemPool_t real = h ? (cuDeviceGetMemPool_t)dlsym(h, "cuDeviceGetMemPool") : NULL;
    return real ? real(pool, dev) : CUDA_ERROR_UNKNOWN;
}

VISIBLE CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    if (getenv("HAMI_TEST_FAIL_NEXT_POOLATTR")) {
        unsetenv("HAMI_TEST_FAIL_NEXT_POOLATTR");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    void *h = real_driver_handle();
    cuMemPoolGetAttribute_t real = h ? (cuMemPoolGetAttribute_t)dlsym(h, "cuMemPoolGetAttribute") : NULL;
    return real ? real(pool, attr, value) : CUDA_ERROR_UNKNOWN;
}

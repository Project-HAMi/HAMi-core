// LD_PRELOAD companion that forces one-shot failures out of the real CUDA
// driver, for exercising the failure branches added in the #312 fix
// (src/cuda/memory.c: cuMemPoolGetAttribute, cuMemGetInfo_v2,
// cuMemAddressReserve; src/cuda/device.c: cuDeviceGetCount).
//
// Mechanism (same trick as test/fault_inject_driver.c from PR #307): it does
// NOT replace libcuda.so.1 -- it interposes dlopen() so that when libvgpu.so
// (the wrapper under test) dlopen()s "libcuda.so.1" to build its
// cuda_library_entry dispatch table, it gets a handle to *this* library
// instead. This library exports exactly four faked symbols; dlsym(table,
// name) finds those here, and forwards every other symbol lookup to the real
// driver via a saved real dlopen(). Nothing outside these four entry points
// is touched.
//
// Load order matters: LD_PRELOAD="fault_inject_driver_312.so:libvgpu.so" --
// this library first so it observes libvgpu.so's dlopen("libcuda.so.1").
//
// Env vars (each is one-shot: read once, then unset by the fake itself):
//   HAMI_TEST_FAIL_NEXT_POOLATTR    -> next cuMemPoolGetAttribute fails
//   HAMI_TEST_FAIL_NEXT_MEMGETINFO  -> next cuMemGetInfo_v2 fails
//   HAMI_TEST_FAIL_NEXT_ADDRRESERVE -> next cuMemAddressReserve fails
//   HAMI_TEST_FAIL_NEXT_DEVCOUNT    -> next cuDeviceGetCount fails
//
// Needs a real GPU to build/run against a real libcuda.so.1; not exercised
// in CI (this project's CI only compiles, see .github/workflows/build-src.yml
// and Makefile's build-in-docker target -- no GPU runner is available there).

#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) && __GNUC__ >= 4
#define VISIBLE __attribute__((visibility("default")))
#else
#define VISIBLE
#endif

typedef void *(*dlopen_t)(const char *, int);
typedef CUresult (*cuMemPoolGetAttribute_t)(CUmemoryPool, CUmemPool_attribute, void *);
typedef CUresult (*cuMemGetInfo_v2_t)(size_t *, size_t *);
typedef CUresult (*cuMemAddressReserve_t)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
typedef CUresult (*cuDeviceGetCount_t)(int *);

static void *g_real_driver_handle = NULL;

static dlopen_t real_dlopen(void) {
    static dlopen_t fn = NULL;
    if (!fn)
        fn = (dlopen_t)dlsym(RTLD_NEXT, "dlopen");
    return fn;
}

static void *self_handle(void) {
    Dl_info info;
    if (dladdr((void *)self_handle, &info))
        return real_dlopen()(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    return NULL;
}

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
                    "[fault_inject_driver_312] substituting for libcuda.so.1: "
                    "only cuMemPoolGetAttribute/cuMemGetInfo_v2/"
                    "cuMemAddressReserve/cuDeviceGetCount are faked, "
                    "everything else falls back to the real driver\n");
            return self;
        }
    }
    return real_result;
}

VISIBLE CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    if (getenv("HAMI_TEST_FAIL_NEXT_POOLATTR")) {
        unsetenv("HAMI_TEST_FAIL_NEXT_POOLATTR");
        return CUDA_ERROR_INVALID_VALUE;
    }
    void *h = real_driver_handle();
    cuMemPoolGetAttribute_t real = h ? (cuMemPoolGetAttribute_t)dlsym(h, "cuMemPoolGetAttribute") : NULL;
    return real ? real(pool, attr, value) : CUDA_ERROR_UNKNOWN;
}

VISIBLE CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    if (getenv("HAMI_TEST_FAIL_NEXT_MEMGETINFO")) {
        unsetenv("HAMI_TEST_FAIL_NEXT_MEMGETINFO");
        return CUDA_ERROR_UNKNOWN;
    }
    void *h = real_driver_handle();
    cuMemGetInfo_v2_t real = h ? (cuMemGetInfo_v2_t)dlsym(h, "cuMemGetInfo_v2") : NULL;
    return real ? real(free, total) : CUDA_ERROR_UNKNOWN;
}

VISIBLE CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                      CUdeviceptr addr, unsigned long long flags) {
    if (getenv("HAMI_TEST_FAIL_NEXT_ADDRRESERVE")) {
        unsetenv("HAMI_TEST_FAIL_NEXT_ADDRRESERVE");
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    void *h = real_driver_handle();
    cuMemAddressReserve_t real = h ? (cuMemAddressReserve_t)dlsym(h, "cuMemAddressReserve") : NULL;
    return real ? real(ptr, size, alignment, addr, flags) : CUDA_ERROR_UNKNOWN;
}

VISIBLE CUresult cuDeviceGetCount(int *count) {
    if (getenv("HAMI_TEST_FAIL_NEXT_DEVCOUNT")) {
        unsetenv("HAMI_TEST_FAIL_NEXT_DEVCOUNT");
        return CUDA_ERROR_UNKNOWN;
    }
    void *h = real_driver_handle();
    cuDeviceGetCount_t real = h ? (cuDeviceGetCount_t)dlsym(h, "cuDeviceGetCount") : NULL;
    return real ? real(count) : CUDA_ERROR_UNKNOWN;
}

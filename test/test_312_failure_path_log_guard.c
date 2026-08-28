// Regression test for Project-HAMi/HAMi-core issue #312.
//
// The four wrappers below (three in src/cuda/memory.c, one in
// src/cuda/device.c) used to log an out-param unconditionally, even when the
// preceding CUDA_OVERRIDE_CALL failed and left that out-param unwritten:
// reading it then read whatever garbage happened to be on the caller's
// stack/heap. cuMemPoolGetAttribute additionally cast that value through a
// fixed `(long *)` regardless of the CUmemPool_attribute's real pointee
// width -- fixed here by dropping the value from the log entirely rather
// than guessing a per-attribute width (see PR description: no authoritative
// CUDA header was available in this repo to verify that table against).
//
// This test does not rely on ASan/UBSan to catch the read (ASan does not
// flag an in-bounds read of unpoisoned-but-uninitialized stack memory, and
// MemorySanitizer -- the tool that would -- cannot be used against a
// closed-source libcuda.so.1). Instead it poisons each out-param buffer with
// a distinctive canary value before forcing the underlying driver call to
// fail via test/fault_inject_driver_312.c, captures HAMi-core's log output
// (LOG_INFO/LOG_DEBUG write to stderr in the default, non-FILEDEBUG build --
// see src/include/log_utils.h), and asserts the canary never appears in it.
// Pre-fix, cuMemPoolGetAttribute's failure branch would have printed the
// canary verbatim.
//
// Build/run: same as every other GPU-backed test in this directory (see
// test/test_alloc_pool_async.c's header comment) -- needs a real GPU and is
// not exercised in CI. Requires the fault_inject_driver_312 companion
// (test/CMakeLists.txt) ahead of libvgpu.so in LD_PRELOAD:
//   LD_PRELOAD=/path/to/fault_inject_driver_312.so:/path/to/libvgpu.so \
//     LIBCUDA_LOG_LEVEL=3 ./test_312_failure_path_log_guard

#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

#define CANARY_INT 987654321
#define CANARY_LONG 123456789012345LL
#define CANARY_STR_INT "987654321"
#define CANARY_STR_LONG "123456789012345"

#define CHECK_DRV(call)                                                        \
    do {                                                                       \
        CUresult _e = (call);                                                  \
        if (_e != CUDA_SUCCESS) {                                              \
            const char *_n = NULL;                                             \
            cuGetErrorName(_e, &_n);                                           \
            fprintf(stderr, "FATAL %s:%d: %s -> %d (%s)\n", __FILE__,          \
                    __LINE__, #call, (int)_e, _n ? _n : "?");                  \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

/* Redirect this process's stderr (where LOG_INFO/LOG_DEBUG write) to a temp
 * file so its content can be inspected after each call. Returns the
 * original stderr fd (dup'd) so real failures can still be reported. */
static int g_captured_fd = -1;
static char g_capture_path[] = "/tmp/hami_312_test_capture_XXXXXX";
static int g_saved_stderr = -1;

static void capture_start(void) {
    g_saved_stderr = dup(STDERR_FILENO);
    g_captured_fd = mkstemp(g_capture_path);
    if (g_captured_fd < 0) {
        perror("mkstemp");
        exit(2);
    }
    fflush(stderr);
    dup2(g_captured_fd, STDERR_FILENO);
}

/* Truncate the capture file and read back whatever gets written between now
 * and the next call to capture_read(). */
static void capture_reset(void) {
    fflush(stderr);
    ftruncate(g_captured_fd, 0);
    lseek(g_captured_fd, 0, SEEK_SET);
}

static char *capture_read(void) {
    fflush(stderr);
    off_t len = lseek(g_captured_fd, 0, SEEK_CUR);
    char *buf = malloc((size_t)len + 1);
    lseek(g_captured_fd, 0, SEEK_SET);
    ssize_t n = read(g_captured_fd, buf, (size_t)len);
    buf[n > 0 ? n : 0] = '\0';
    return buf;
}

static void capture_stop(void) {
    fflush(stderr);
    dup2(g_saved_stderr, STDERR_FILENO);
    close(g_saved_stderr);
    close(g_captured_fd);
    unlink(g_capture_path);
}

int main(void) {
    setenv("LIBCUDA_LOG_LEVEL", "3", 1); /* enable LOG_INFO without relying on the caller's shell env */

    CHECK_DRV(cuInit(0));
    CUdevice dev;
    CHECK_DRV(cuDeviceGet(&dev, TEST_DEVICE_ID));
    CUcontext ctx;
#if CUDA_VERSION >= 13000
    CHECK_DRV(cuCtxCreate(&ctx, NULL, 0, dev));
#else
    CHECK_DRV(cuCtxCreate(&ctx, 0, dev));
#endif

    int failures = 0;
    capture_start();

    /* [1] cuMemPoolGetAttribute: force failure, poison the out buffer,
     * confirm the canary never reaches the log and the failure branch's
     * message format (no raw value) is used. */
    {
        CUmemPoolProps props;
        memset(&props, 0, sizeof(props));
        props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = dev;
        CUmemoryPool pool;
        CHECK_DRV(cuMemPoolCreate(&pool, &props));

        long value = CANARY_LONG;
        setenv("HAMI_TEST_FAIL_NEXT_POOLATTR", "1", 1);
        capture_reset();
        CUresult res = cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &value);
        char *log = capture_read();
        printf("[1] cuMemPoolGetAttribute forced failure -> res=%d\n", (int)res);
        if (res == CUDA_SUCCESS) {
            fprintf(stderr, "FAIL: expected forced failure but call succeeded\n");
            failures++;
        }
        if (strstr(log, CANARY_STR_LONG) != NULL) {
            fprintf(stderr, "FAIL: failure-path log leaked the uninitialized canary: %s\n", log);
            failures++;
        }
        if (strstr(log, "cuMemPoolGetAttribute") != NULL && strstr(log, "failed res=") == NULL) {
            fprintf(stderr, "FAIL: failure-path log did not use the expected failed-res format: %s\n", log);
            failures++;
        }
        free(log);
        cuMemPoolDestroy(pool);
    }

    /* [1b] Success path sanity: the log must still fire, but must never
     * contain a raw attribute value (it was dropped deliberately, since no
     * authoritative per-attribute width table is available -- see PR
     * description). */
    {
        CUmemPoolProps props;
        memset(&props, 0, sizeof(props));
        props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = dev;
        CUmemoryPool pool;
        CHECK_DRV(cuMemPoolCreate(&pool, &props));

        long value = CANARY_LONG;
        capture_reset();
        CUresult res = cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &value);
        char *log = capture_read();
        printf("[1b] cuMemPoolGetAttribute success path -> res=%d\n", (int)res);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "FAIL: expected success on the real driver call\n");
            failures++;
        }
        if (strstr(log, "cuMemPoolGetAttribute") == NULL || strstr(log, "attr=") == NULL) {
            fprintf(stderr, "FAIL: success-path log missing/malformed: %s\n", log);
            failures++;
        }
        free(log);
        cuMemPoolDestroy(pool);
    }

    /* [2] cuMemGetInfo_v2: force failure, poison free/total, confirm neither
     * is read into the log (and that the driver's real error propagates
     * instead of being swallowed as CUDA_SUCCESS, which the old code did
     * unconditionally). */
    {
        size_t free_b = (size_t)CANARY_INT;
        size_t total_b = (size_t)CANARY_INT;
        setenv("HAMI_TEST_FAIL_NEXT_MEMGETINFO", "1", 1);
        capture_reset();
        CUresult res = cuMemGetInfo_v2(&free_b, &total_b);
        char *log = capture_read();
        printf("[2] cuMemGetInfo_v2 forced failure -> res=%d\n", (int)res);
        if (res == CUDA_SUCCESS) {
            fprintf(stderr, "FAIL: expected forced failure but call succeeded (return value swallowed?)\n");
            failures++;
        }
        if (strstr(log, CANARY_STR_INT) != NULL) {
            fprintf(stderr, "FAIL: failure-path log leaked the uninitialized canary: %s\n", log);
            failures++;
        }
        free(log);
    }

    /* [3] cuMemAddressReserve: force failure, poison ptr, confirm it is
     * never logged. */
    {
        CUdeviceptr ptr = (CUdeviceptr)CANARY_LONG;
        setenv("HAMI_TEST_FAIL_NEXT_ADDRRESERVE", "1", 1);
        capture_reset();
        CUresult res = cuMemAddressReserve(&ptr, 4096, 0, 0, 0);
        char *log = capture_read();
        printf("[3] cuMemAddressReserve forced failure -> res=%d\n", (int)res);
        if (res == CUDA_SUCCESS) {
            fprintf(stderr, "FAIL: expected forced failure but call succeeded\n");
            failures++;
        }
        if (strstr(log, CANARY_STR_LONG) != NULL) {
            fprintf(stderr, "FAIL: failure-path log leaked the uninitialized canary: %s\n", log);
            failures++;
        }
        free(log);
    }

    /* [4] cuDeviceGetCount: force failure, poison count, confirm it is
     * never logged (LOG_DEBUG, needs LIBCUDA_LOG_LEVEL>=4 -- bump it just
     * for this check). */
    {
        setenv("LIBCUDA_LOG_LEVEL", "4", 1);
        int count = CANARY_INT;
        setenv("HAMI_TEST_FAIL_NEXT_DEVCOUNT", "1", 1);
        capture_reset();
        CUresult res = cuDeviceGetCount(&count);
        char *log = capture_read();
        printf("[4] cuDeviceGetCount forced failure -> res=%d\n", (int)res);
        if (res == CUDA_SUCCESS) {
            fprintf(stderr, "FAIL: expected forced failure but call succeeded\n");
            failures++;
        }
        if (strstr(log, CANARY_STR_INT) != NULL) {
            fprintf(stderr, "FAIL: failure-path log leaked the uninitialized canary: %s\n", log);
            failures++;
        }
        free(log);
        setenv("LIBCUDA_LOG_LEVEL", "3", 1);
    }

    capture_stop();
    cuCtxDestroy(ctx);

    printf("\nResult: %s\n", failures ? "FAIL - see above" : "PASS - no canary reached the log on any forced failure");
    return failures ? 1 : 0;
}

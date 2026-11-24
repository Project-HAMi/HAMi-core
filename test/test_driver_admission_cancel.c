/*
 * GPU-free test: a thread cancelled inside a driver call that holds an
 * admission lease must still give the lease back.
 *
 * It links the production cuInit and primary context hooks and the broker
 * client, installs its own driver entry table and runs a fake broker that
 * holds each lease until EOF. The fake driver blocks in pause(), which is a
 * cancellation point, so the cancel lands inside the driver call.
 *
 * It also checks that cuInit skips the permit after one success (until a
 * fork), that retain skips it while the primary context is active, that a
 * non-final release skips it, that a reset of an active context takes the
 * release permit and keeps the retain count, that host PID detection under the fallback
 * lock never asks for one, that a missing permit warns once per process,
 * and that after one timed-out wait the process stops asking until a fork.
 */
#undef NDEBUG
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "include/libvgpu.h"
#include "include/libcuda_hook.h"

/* Driver entry table and the state the hooks touch. */
cuda_entry_t cuda_library_entry[CUDA_ENTRY_END];
size_t context_size = 1;
int ctx_activate[16];
int pidfound;
int env_utilization_switch;
extern pthread_once_t pre_cuinit_flag;
void childReinitPostInit(void);

void load_cuda_libraries(void) {}
void ensure_initialized(void) {}
void size_primary_context_lazily(int cudadev) { (void)cudadev; }
void reset_lazy_context_sizing(void) {}
void graph_primary_context(CUdevice dev, CUcontext ctx) { (void)dev; (void)ctx; }
void graph_primary_context_prepare_release(CUdevice dev) { (void)dev; }
void graph_primary_context_released(CUdevice dev, int reset) {
    (void)dev; (void)reset;
}
int add_gpu_device_memory_usage(int32_t pid, int dev, size_t usage,
                                int type) {
    (void)pid; (void)dev; (void)usage; (void)type;
    return 0;
}
static sem_t driver_entered;
static volatile int driver_blocks;
static volatile int accounting_blocks;

/* Blocks like the fake driver, so a cancel can land in the usage update. */
static void block_here(void) {
    sem_post(&driver_entered);
    for (;;) {
        pause();
    }
}

int rm_gpu_device_memory_usage(int32_t pid, int dev, size_t usage, int type) {
    (void)pid; (void)dev; (void)usage; (void)type;
    if (accounting_blocks) {
        block_here();
    }
    return 0;
}

/* postInit and host PID dependencies (cuInit and ensure_hostpid reach them). */
/* fallback_mode makes host PID detection take the fallback path, where
 * set_task_pid retains and releases a primary context under the lock. */
static volatile int fallback_mode;
static volatile int fallback_lock_held;
static volatile int fallback_probes;

void allocator_init(void) {}
void print_all(void) {}
int map_cuda_visible_devices(void) { return 0; }
nvmlReturn_t set_task_pid(void) {
    CUcontext context;

    if (fallback_mode) {
        assert(fallback_lock_held);
        assert(cuDevicePrimaryCtxRetain(&context, 0) == CUDA_SUCCESS);
        assert(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
        fallback_probes++;
    }
    return NVML_SUCCESS;
}
nvmlReturn_t set_task_pid_from_broker(void) {
    return fallback_mode ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
}
int lock_postinit(void) { return 1; }
void unlock_postinit(void) {}
int set_env_utilization_switch(void) { return 0; }
void init_utilization_watcher(void) {}
int hostpid_fallback_lock_deadline_after_ms(struct timespec *deadline,
                                           unsigned int timeout_ms) {
    (void)deadline; (void)timeout_ms;
    return fallback_mode ? 0 : -1;
}
int hostpid_fallback_lock_acquire_until(const struct timespec *deadline) {
    (void)deadline;
    if (!fallback_mode) {
        return -1;
    }
    fallback_lock_held = 1;
    return 0;
}
int hostpid_fallback_lock_release(void) {
    fallback_lock_held = 0;
    return 0;
}
void hostpid_fallback_lock_after_fork(void) {}

static char socket_path[PATH_MAX];
static volatile int admission_requests;
static volatile int primary_active;

/* The trusted broker lives at a root-owned path; use the test socket. */
int __wrap_hostpid_broker_acquire_driver_admission_trusted(
    hostpid_driver_admission_operation_t operation, int *lease_fd) {
    admission_requests++;
    return hostpid_broker_acquire_driver_admission(socket_path, operation,
                                                   lease_fd);
}

static CUresult fake_driver(void) {
    if (driver_blocks) {
        block_here();
    }
    return CUDA_SUCCESS;
}
static CUresult fake_init(unsigned int flags) {
    (void)flags;
    return fake_driver();
}
static CUresult fake_retain(CUcontext *context, CUdevice device) {
    (void)context; (void)device;
    return fake_driver();
}
static CUresult fake_release(CUdevice device) {
    (void)device;
    return fake_driver();
}
static CUresult fake_reset(CUdevice device) {
    (void)device;
    return fake_driver();
}
static CUresult fake_get_state(CUdevice device, unsigned int *flags,
                               int *active) {
    (void)device;
    *flags = 0;
    *active = primary_active;
    return CUDA_SUCCESS;
}

static void noop(void) {}

/* call_hook value for cuDevicePrimaryCtxReset_v2, which sends command 4. */
#define RESET_CASE 100

static void *call_hook(void *argument) {
    CUcontext context;

    switch (*(int *)argument) {
    case HOSTPID_DRIVER_ADMISSION_INIT:
        cuInit(0);
        break;
    case HOSTPID_DRIVER_ADMISSION_RETAIN:
        cuDevicePrimaryCtxRetain(&context, 0);
        break;
    case RESET_CASE:
        cuDevicePrimaryCtxReset_v2(0);
        break;
    default:
        cuDevicePrimaryCtxRelease_v2(0);
        break;
    }
    return NULL;
}

/* Grants one permit, then exits 0 only when the client closes the lease. */
static pid_t start_broker(unsigned char command) {
    struct sockaddr_un address;
    char directory[] = "/tmp/hami-admission-cancel-XXXXXX";
    int listener = socket(AF_UNIX, SOCK_STREAM, 0);
    pid_t broker;

    assert(listener >= 0);
    assert(mkdtemp(directory) != NULL);
    snprintf(socket_path, sizeof(socket_path), "%s/broker.sock", directory);
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, socket_path);
    assert(bind(listener, (struct sockaddr *)&address, sizeof(address)) == 0);
    assert(listen(listener, 1) == 0);

    broker = fork();
    assert(broker >= 0);
    if (broker == 0) {
        const unsigned char expected[8] = {'H', 'P', 'I', 'D', 0, 1, 0,
                                           command};
        const unsigned char granted[12] = {'H', 'P', 'I', 'D', 0, 1, 0, 0,
                                           0, 0, 0, 1};
        unsigned char request[8];
        unsigned char byte;
        size_t offset = 0;
        int connection;

        alarm(5);
        connection = accept(listener, NULL, NULL);
        assert(connection >= 0);
        while (offset < sizeof(request)) {
            ssize_t received = read(connection, request + offset,
                                    sizeof(request) - offset);
            assert(received > 0);
            offset += (size_t)received;
        }
        assert(memcmp(request, expected, sizeof(request)) == 0);
        assert(write(connection, granted, sizeof(granted)) ==
               (ssize_t)sizeof(granted));
        _exit(read(connection, &byte, 1) == 0 ? 0 : 1);
    }
    close(listener);
    return broker;
}

static void finish_broker(pid_t broker) {
    char directory[PATH_MAX];
    int status;

    assert(waitpid(broker, &status, 0) == broker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    snprintf(directory, sizeof(directory), "%s", socket_path);
    *strrchr(directory, '/') = '\0';
    assert(unlink(socket_path) == 0);
    assert(rmdir(directory) == 0);
}

static void run_case(int operation, int cancel, int in_accounting) {
    int command = operation == RESET_CASE ? HOSTPID_DRIVER_ADMISSION_RELEASE
                                          : operation;
    pid_t broker = start_broker((unsigned char)(command + 2));
    pthread_t thread;
    void *result;

    ctx_activate[0] = 1;
    driver_blocks = cancel && !in_accounting;
    accounting_blocks = cancel && in_accounting;
    assert(pthread_create(&thread, NULL, call_hook, &operation) == 0);
    if (cancel) {
        while (sem_wait(&driver_entered) != 0) {
            assert(errno == EINTR);
        }
        assert(pthread_cancel(thread) == 0);
    }
    assert(pthread_join(thread, &result) == 0);
    assert(result == (cancel ? PTHREAD_CANCELED : NULL));
    // The broker exits 0 only after it reads EOF on the lease.
    finish_broker(broker);
}

/* No permit request reaches the broker; a request would fail anyway. */
static void expect_no_permit(int operation) {
    CUcontext context;
    int before = admission_requests;

    if (operation == HOSTPID_DRIVER_ADMISSION_INIT) {
        assert(cuInit(0) == CUDA_SUCCESS);
    } else if (operation == HOSTPID_DRIVER_ADMISSION_RETAIN) {
        assert(cuDevicePrimaryCtxRetain(&context, 0) == CUDA_SUCCESS);
    } else if (operation == RESET_CASE) {
        assert(cuDevicePrimaryCtxReset_v2(0) == CUDA_SUCCESS);
    } else {
        assert(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS);
    }
    assert(admission_requests == before);
}

static void run_skip_cases(void) {
    int operation = HOSTPID_DRIVER_ADMISSION_INIT;
    pid_t child;
    int status;

    // The first successful cuInit takes a permit, later calls do not.
    run_case(operation, 0, 0);
    expect_no_permit(operation);
    puts("cuInit: no permit after one success");

    // A forked child must initialize again, so it asks for a permit.
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        int before = admission_requests;

        cuInit(0);
        _exit(admission_requests == before + 1 ? 0 : 1);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    puts("cuInit: forked child asks for a permit");

    // The retain cases in main ran with an inactive context and took one.
    primary_active = 1;
    expect_no_permit(HOSTPID_DRIVER_ADMISSION_RETAIN);
    puts("cuDevicePrimaryCtxRetain: no permit while the context is active");

    // Two references: the first release only counts down, the last one
    // can destroy the context and takes a permit.
    expect_no_permit(HOSTPID_DRIVER_ADMISSION_RETAIN);

    // A forked child holds none of the parent's references, so its first
    // release counts as final and asks for a permit.
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        int before = admission_requests;

        cuDevicePrimaryCtxRelease_v2(0);
        _exit(admission_requests == before + 1 ? 0 : 1);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

    expect_no_permit(HOSTPID_DRIVER_ADMISSION_RELEASE);
    primary_active = 0;
    run_case(HOSTPID_DRIVER_ADMISSION_RELEASE, 0, 0);
    puts("cuDevicePrimaryCtxRelease_v2: no permit while a reference stays");

    // The driver keeps its retain count across a reset, so with two
    // references the first release after it only counts down. A second
    // reset finds the context inactive and skips the permit.
    primary_active = 1;
    expect_no_permit(HOSTPID_DRIVER_ADMISSION_RETAIN);
    expect_no_permit(HOSTPID_DRIVER_ADMISSION_RETAIN);
    run_case(RESET_CASE, 0, 0);
    primary_active = 0;
    expect_no_permit(RESET_CASE);
    expect_no_permit(HOSTPID_DRIVER_ADMISSION_RELEASE);
    run_case(HOSTPID_DRIVER_ADMISSION_RELEASE, 0, 0);
    puts("cuDevicePrimaryCtxReset_v2: permit only while active, keeps the count");

    // Host PID detection runs again in a forked child. On the fallback path
    // it retains and releases under the fallback lock without a permit.
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        int before = admission_requests;

        fallback_mode = 1;
        ensure_hostpid();
        _exit(fallback_probes == 1 && !fallback_lock_held &&
              admission_requests == before ? 0 : 1);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    puts("host PID fallback: no permit under the fallback lock");

    // The broker socket is gone, so every request fails. A process warns
    // once; a forked child warns once again.
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        FILE *log = tmpfile();
        char line[256];
        pid_t grandchild;
        int warnings = 0;

        assert(log != NULL && dup2(fileno(log), STDERR_FILENO) >= 0);
        g_log_level = 2;
        assert(driver_admission_acquire(HOSTPID_DRIVER_ADMISSION_INIT) == -1);
        assert(driver_admission_acquire(HOSTPID_DRIVER_ADMISSION_INIT) == -1);
        grandchild = fork();
        assert(grandchild >= 0);
        if (grandchild == 0) {
            _exit(driver_admission_acquire(HOSTPID_DRIVER_ADMISSION_INIT) == -1
                      ? 0 : 1);
        }
        assert(waitpid(grandchild, &status, 0) == grandchild);
        rewind(log);
        while (fgets(line, sizeof(line), log) != NULL) {
            warnings += strstr(line, "Warn(") != NULL &&
                        strstr(line, "No driver admission permit") != NULL;
        }
        _exit(warnings == 2 ? 0 : 1);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    puts("driver admission: missing permit logged once per process");

    // A listener that never accepts is a silent broker: connect and the
    // request write succeed, the reply never comes.
    child = fork();
    assert(child >= 0);
    if (child == 0) {
        struct sockaddr_un address;
        char directory[] = "/tmp/hami-admission-silent-XXXXXX";
        int listener = socket(AF_UNIX, SOCK_STREAM, 0);
        int before;
        pid_t grandchild;

        // alarm() does not survive fork; do not hang on the default wait.
        alarm(10);
        assert(listener >= 0);
        assert(mkdtemp(directory) != NULL);
        memset(&address, 0, sizeof(address));
        address.sun_family = AF_UNIX;
        snprintf(socket_path, sizeof(socket_path), "%s/broker.sock",
                 directory);
        strcpy(address.sun_path, socket_path);
        assert(bind(listener, (struct sockaddr *)&address,
                    sizeof(address)) == 0);
        assert(listen(listener, 8) == 0);
        before = admission_requests;
        errno = 0;
        assert(driver_admission_acquire(HOSTPID_DRIVER_ADMISSION_INIT) == -1);
        assert(errno == ETIMEDOUT && admission_requests == before + 1);
        // After a timeout the process does not ask again.
        assert(driver_admission_acquire(HOSTPID_DRIVER_ADMISSION_INIT) == -1);
        assert(admission_requests == before + 1);
        grandchild = fork();
        assert(grandchild >= 0);
        if (grandchild == 0) {
            driver_admission_acquire(HOSTPID_DRIVER_ADMISSION_INIT);
            _exit(admission_requests == before + 2 ? 0 : 1);
        }
        assert(waitpid(grandchild, &status, 0) == grandchild);
        assert(unlink(socket_path) == 0 && rmdir(directory) == 0);
        _exit(WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    puts("driver admission: no request after a timed-out wait until a fork");
}

int main(void) {
    static const char *names[] = {"cuInit", "cuDevicePrimaryCtxRetain",
                                  "cuDevicePrimaryCtxRelease_v2"};

    alarm(20);
    setbuf(stdout, NULL);
    assert(setenv("LIBCUDA_LOG_LEVEL", "0", 1) == 0);
    assert(setenv("LIBVGPU_HOSTPID_BROKER", "1", 1) == 0);
    assert(setenv("LIBVGPU_DRIVER_ADMISSION", "1", 1) == 0);
    log_utils_init();
    assert(sem_init(&driver_entered, 0, 0) == 0);
    // preInit loads the real libcuda; mark it as done, but register its
    // fork handlers so the fork case runs the production child handler.
    pthread_once(&pre_cuinit_flag, noop);
    assert(pthread_atfork(hostpid_broker_admission_fork_prepare,
                          hostpid_broker_admission_fork_parent,
                          childReinitPostInit) == 0);
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuInit)].fn_ptr = (void *)fake_init;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRetain)].fn_ptr =
        (void *)fake_retain;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRelease_v2)]
        .fn_ptr = (void *)fake_release;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxGetState)].fn_ptr =
        (void *)fake_get_state;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxReset_v2)].fn_ptr =
        (void *)fake_reset;

    for (int operation = HOSTPID_DRIVER_ADMISSION_INIT;
         operation <= HOSTPID_DRIVER_ADMISSION_RELEASE; operation++) {
        // Retain/release run the normal path first, so the cancel case also
        // proves the normal release is not repeated by a cleanup handler.
        if (operation != HOSTPID_DRIVER_ADMISSION_INIT) {
            run_case(operation, 0, 0);
        }
        run_case(operation, 1, 0);
        if (operation == HOSTPID_DRIVER_ADMISSION_RELEASE) {
            // A cancel in the usage update after the call still unlocks.
            run_case(operation, 1, 1);
        }
        printf("%s: cancelled call returns its lease\n", names[operation]);
    }
    primary_active = 1;
    run_case(RESET_CASE, 0, 0);
    run_case(RESET_CASE, 1, 0);
    primary_active = 0;
    puts("cuDevicePrimaryCtxReset_v2: cancelled call returns its lease");
    run_skip_cases();
    puts("driver admission cancel tests passed");
    return 0;
}

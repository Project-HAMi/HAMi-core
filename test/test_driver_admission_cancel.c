/*
 * GPU-free test: a thread cancelled inside a driver call that holds an
 * admission lease must still give the lease back.
 *
 * It links the production cuInit and primary context hooks and the broker
 * client, installs its own driver entry table and runs a fake broker that
 * holds each lease until EOF. The fake driver blocks in pause(), which is a
 * cancellation point, so the cancel lands inside the driver call.
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

void load_cuda_libraries(void) {}
void ensure_initialized(void) {}
void size_primary_context_lazily(int cudadev) { (void)cudadev; }
void reset_lazy_context_sizing(void) {}
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

/* postInit dependencies; a cancelled cuInit never reaches them. */
void allocator_init(void) {}
void print_all(void) {}
int map_cuda_visible_devices(void) { return 0; }
nvmlReturn_t set_task_pid(void) { return NVML_SUCCESS; }
nvmlReturn_t set_task_pid_from_broker(void) { return NVML_SUCCESS; }
int lock_postinit(void) { return 1; }
void unlock_postinit(void) {}
int set_env_utilization_switch(void) { return 0; }
void init_utilization_watcher(void) {}
int hostpid_fallback_lock_deadline_after_ms(struct timespec *deadline,
                                           unsigned int timeout_ms) {
    (void)deadline; (void)timeout_ms;
    return -1;
}
int hostpid_fallback_lock_acquire_until(const struct timespec *deadline) {
    (void)deadline;
    return -1;
}
int hostpid_fallback_lock_release(void) { return 0; }
void hostpid_fallback_lock_after_fork(void) {}

static char socket_path[PATH_MAX];

/* The trusted broker lives at a root-owned path; use the test socket. */
int __wrap_hostpid_broker_acquire_driver_admission_trusted(
    hostpid_driver_admission_operation_t operation, int *lease_fd) {
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

static void noop(void) {}

static void *call_hook(void *argument) {
    CUcontext context;

    switch (*(int *)argument) {
    case HOSTPID_DRIVER_ADMISSION_INIT:
        cuInit(0);
        break;
    case HOSTPID_DRIVER_ADMISSION_RETAIN:
        cuDevicePrimaryCtxRetain(&context, 0);
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
    pid_t broker = start_broker((unsigned char)(operation + 2));
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
    // preInit loads the real libcuda; mark it as done.
    pthread_once(&pre_cuinit_flag, noop);
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuInit)].fn_ptr = (void *)fake_init;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRetain)].fn_ptr =
        (void *)fake_retain;
    cuda_library_entry[CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRelease_v2)]
        .fn_ptr = (void *)fake_release;

    for (int operation = HOSTPID_DRIVER_ADMISSION_INIT;
         operation <= HOSTPID_DRIVER_ADMISSION_RELEASE; operation++) {
        // Retain/release run the normal path first, so the cancel case also
        // proves the normal release is not repeated by a cleanup handler.
        if (operation != HOSTPID_DRIVER_ADMISSION_INIT) {
            run_case(operation, 0, 0);
        }
        run_case(operation, 1, 0);
        if (operation == HOSTPID_DRIVER_ADMISSION_RELEASE) {
            // Release holds its permit across the usage update too.
            run_case(operation, 1, 1);
        }
        printf("%s: cancelled call returns its lease\n", names[operation]);
    }
    puts("driver admission cancel tests passed");
    return 0;
}

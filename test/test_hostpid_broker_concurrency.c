#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "include/libvgpu.h"
#include "include/libcuda_hook.h"
#include "include/hostpid_fallback_lock.h"
#include "multiprocess/multiprocess_memory_limit.h"

int pidfound;
int env_utilization_switch;
size_t context_size;
int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

static char node_path[] = "/tmp/hami-broker-concurrency.XXXXXX";
static char cache_path[] = "/tmp/hami-broker-cache.XXXXXX";
static int cache_fd;
static int broker_actor;
static int broker_attempted;
static struct {
    sem_t fallback_window;
    sem_t broker_attempt;
    sem_t fallback_continue;
    sem_t fallback_done;
    int active[2];
    pid_t pid[2];
    pid_t saved[2];
} *state;

static void require(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL %s (errno=%d)\n", message, errno);
        exit(1);
    }
}

static void wait_sem(sem_t *sem) {
    while (sem_wait(sem) != 0) {
        require(errno == EINTR, "wait for test barrier");
    }
}

static CUresult device_count(int *count) {
    *count = 1;
    return CUDA_SUCCESS;
}

cuda_entry_t cuda_library_entry[CUDA_ENTRY_END] = {
    [OVERRIDE_cuDeviceGetCount] = {.fn_ptr = device_count},
};

void allocator_init(void) {}
void init_utilization_watcher(void) {}
int set_env_utilization_switch(void) { return 0; }
int set_host_pid(int pid) { state->saved[broker_actor] = pid; return 0; }

// Only the broker reply and driver boundary are simulated. postInit, both
// lookup paths and the shared/exclusive node lock run their production code.
int __wrap_hostpid_broker_query_trusted(const char *path, pid_t *pid) {
    (void)path;
    if (broker_actor) {
        *pid = getpid();
        return 0;
    }
    *pid = 0;
    errno = ETIMEDOUT;
    return -1;
}

int __wrap_hostpid_fallback_lock_acquire_until(const struct timespec *deadline) {
    return hostpid_fallback_lock_acquire_at_until(node_path, getuid(), deadline);
}

int __wrap_hostpid_fallback_lock_acquire_shared_until(
    const struct timespec *deadline) {
    return hostpid_fallback_lock_acquire_shared_at_until(
        node_path, getuid(), deadline);
}

static void signal_broker_attempt(void) {
    if (!broker_attempted) {
        broker_attempted = 1;
        require(sem_post(&state->broker_attempt) == 0, "signal broker attempt");
    }
}

int lock_postinit(void) {
    cache_fd = open(cache_path, O_RDWR);
    return cache_fd >= 0 && flock(cache_fd, LOCK_EX) == 0;
}

void unlock_postinit(void) {
    require(flock(cache_fd, LOCK_UN) == 0, "unlock cache");
    require(close(cache_fd) == 0, "close cache lock");
}

unsigned int cuda_to_nvml_map(unsigned int dev) { return dev; }
unsigned int nvml_to_cuda_map(unsigned int dev) { return dev; }
nvmlReturn_t nvmlInit(void) { return NVML_SUCCESS; }
nvmlReturn_t nvmlDeviceGetCount(unsigned int *count) {
    *count = 1;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    (void)index;
    *device = (nvmlDevice_t)1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *count, nvmlProcessInfo_v1_t *processes) {
    (void)device;
    unsigned int required = state->active[0] + state->active[1];
    if (*count < required) {
        *count = required;
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    *count = 0;
    // Listing the broker first exposes a fallback diff polluted by its retain.
    for (int actor = 1; actor >= 0; actor--) {
        if (state->active[actor]) {
            processes[*count].pid = state->pid[actor];
            processes[*count].usedGpuMemory = 1234;
            (*count)++;
        }
    }
    return NVML_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *context, CUdevice device) {
    (void)device;
    *context = (CUcontext)1;
    if (!broker_actor) {
        // The first NVML snapshot is complete and both discovery locks are held.
        require(sem_post(&state->fallback_window) == 0, "signal fallback window");
        wait_sem(&state->fallback_continue);
    }
    state->active[broker_actor] = 1;
    if (broker_actor) {
        // Without the sizing guard this notification comes from inside the
        // context, ensuring the fallback sees both PIDs and fails the assertion.
        signal_broker_attempt();
        wait_sem(&state->fallback_done);
    }
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice device) {
    (void)device;
    state->active[broker_actor] = 0;
    return CUDA_SUCCESS;
}

static void run_child(int is_broker) {
    broker_actor = is_broker;
    state->pid[broker_actor] = getpid();
    if (broker_actor) {
        hostpid_fallback_lock_set_before_flock_hook(signal_broker_attempt);
    }
    ensure_post_init();
    require(hostpid_fallback_lock_active_fd() == -1, "init releases node lock");
    if (!broker_actor) {
        require(sem_post(&state->fallback_done) == 0, "signal fallback completion");
    }
    require(pidfound == 1 && state->saved[broker_actor] == getpid(),
            "each path registers its own PID");
    require(context_size == 1234, "each path sizes its own context");
    _exit(0);
}

static int child_succeeded(pid_t child) {
    int status;
    return waitpid(child, &status, 0) == child && WIFEXITED(status) &&
           WEXITSTATUS(status) == 0;
}

int main(void) {
    alarm(10);
    require(setenv("LIBVGPU_HOSTPID_BROKER", "1", 1) == 0, "enable broker");
    require(unsetenv("CUDA_VISIBLE_DEVICES") == 0, "use identity device mapping");
    require(mkdtemp(node_path) != NULL, "create node lock directory");
    int fixture = mkstemp(cache_path);
    require(fixture >= 0 && close(fixture) == 0, "create cache lock");
    state = mmap(NULL, sizeof(*state), PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_SHARED, -1, 0);
    require(state != MAP_FAILED, "map shared test state");
    require(sem_init(&state->fallback_window, 1, 0) == 0 &&
            sem_init(&state->broker_attempt, 1, 0) == 0 &&
            sem_init(&state->fallback_continue, 1, 0) == 0 &&
            sem_init(&state->fallback_done, 1, 0) == 0, "initialize barriers");

    pid_t fallback = fork();
    require(fallback >= 0, "fork fallback caller");
    if (fallback == 0) run_child(0);
    wait_sem(&state->fallback_window);
    pid_t broker = fork();
    require(broker >= 0, "fork broker caller");
    if (broker == 0) run_child(1);
    wait_sem(&state->broker_attempt);
    require(sem_post(&state->fallback_continue) == 0, "finish fallback snapshot");
    int fallback_ok = child_succeeded(fallback);
    int broker_ok = child_succeeded(broker);
    printf("fallback PID %d saved %d; broker PID %d saved %d\n",
           state->pid[0], state->saved[0], state->pid[1], state->saved[1]);
    sem_destroy(&state->fallback_window);
    sem_destroy(&state->broker_attempt);
    sem_destroy(&state->fallback_continue);
    sem_destroy(&state->fallback_done);
    require(munmap(state, sizeof(*state)) == 0, "unmap test state");
    require(unlink(cache_path) == 0 && rmdir(node_path) == 0, "remove fixtures");
    require(fallback_ok && broker_ok, "mixed broker and fallback initialization");
    puts("host PID broker concurrency tests passed");
    return 0;
}

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/libvgpu.h"
#include "include/hostpid_fallback_lock.h"

void postInit(void);
void childReinitPostInit(void);
extern pthread_once_t post_cuinit_flag;

int pidfound;
int env_utilization_switch;

static int failures;
static char events[512];
static nvmlReturn_t broker_result;
static nvmlReturn_t probe_result;
static int deadline_result;
static int node_lock_errno;
static int cache_lock_result;
static int release_result;
static int node_locked;
static int cache_locked;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL %s (events: %s)\n", message, events);
        failures++;
    }
}

static void record(const char *event) {
    size_t used = strlen(events);
    int written = snprintf(events + used, sizeof(events) - used, "%s ", event);

    if (written < 0 || (size_t)written >= sizeof(events) - used) {
        fprintf(stderr, "event buffer exhausted\n");
        exit(1);
    }
}

void allocator_init(void) {
    record("allocator");
}

int map_cuda_visible_devices(void) {
    record("map");
    return 0;
}

nvmlReturn_t set_task_pid_from_broker(void) {
    check(!node_locked && !cache_locked, "broker query needs no discovery lock");
    record("broker");
    return broker_result;
}

int hostpid_fallback_lock_deadline_after_ms(struct timespec *deadline,
                                           unsigned int timeout_ms) {
    record("deadline");
    check(timeout_ms == HOSTPID_FALLBACK_LOCK_TIMEOUT_MS,
          "fallback uses its configured timeout");
    deadline->tv_sec = 123;
    deadline->tv_nsec = 456;
    errno = EIO;
    return deadline_result;
}

int hostpid_fallback_lock_acquire_until(const struct timespec *deadline) {
    record("node_lock");
    check(!cache_locked, "node lock is acquired before cache lock");
    check(deadline->tv_sec == 123 && deadline->tv_nsec == 456,
          "node lock receives the computed deadline");
    if (node_lock_errno != 0) {
        errno = node_lock_errno;
        return -1;
    }
    node_locked = 1;
    return 0;
}

int lock_postinit(void) {
    record("cache_lock");
    cache_locked = cache_lock_result;
    return cache_lock_result;
}

nvmlReturn_t set_task_pid(void) {
    record("probe");
    check(cache_locked, "NVML discovery holds the cache lock");
    if (getenv("LIBVGPU_HOSTPID_BROKER") != NULL &&
        strcmp(getenv("LIBVGPU_HOSTPID_BROKER"), "1") == 0) {
        check(node_locked, "enabled fallback holds the node lock during NVML");
    }
    return probe_result;
}

void unlock_postinit(void) {
    record("cache_unlock");
    check(cache_locked, "only an acquired cache lock is released");
    cache_locked = 0;
}

int hostpid_fallback_lock_release(void) {
    record("node_unlock");
    check(node_locked, "only an acquired node lock is released");
    check(!cache_locked, "cache lock is released before node lock");
    node_locked = 0;
    errno = EIO;
    return release_result;
}

int set_env_utilization_switch(void) {
    record("utilization");
    return 7;
}

void init_utilization_watcher(void) {
    record("watcher");
    check(!cache_locked && !node_locked, "watcher starts after lock release");
}

void hostpid_fallback_lock_after_fork(void) {
    record("fork_reset");
    node_locked = 0;
}

static void reset(const char *gate) {
    if (gate == NULL) {
        check(unsetenv("LIBVGPU_HOSTPID_BROKER") == 0, "unset broker gate");
    } else {
        check(setenv("LIBVGPU_HOSTPID_BROKER", gate, 1) == 0, "set broker gate");
    }
    events[0] = '\0';
    broker_result = NVML_SUCCESS;
    probe_result = NVML_SUCCESS;
    deadline_result = 0;
    node_lock_errno = 0;
    cache_lock_result = 1;
    release_result = 0;
    node_locked = 0;
    cache_locked = 0;
    pidfound = -1;
    env_utilization_switch = -1;
    post_cuinit_flag = PTHREAD_ONCE_INIT;
}

static void expect(const char *sequence, int found) {
    check(strcmp(events, sequence) == 0, "init uses the expected discovery path");
    check(pidfound == found, "pidfound reflects discovery success");
    check(env_utilization_switch == 7, "initialization keeps utilization settings");
    check(!node_locked && !cache_locked, "initialization leaves no lock held");
}

int main(void) {
    static const char *disabled[] = {NULL, "", "0", "true", "01", " 1"};
    static const char legacy[] =
        "allocator map cache_lock probe cache_unlock utilization watcher ";
    static const char broker[] = "allocator map broker utilization watcher ";
    static const char fallback[] = "allocator map broker deadline node_lock "
        "cache_lock probe cache_unlock node_unlock utilization watcher ";

    for (size_t i = 0; i < sizeof(disabled) / sizeof(disabled[0]); i++) {
        reset(disabled[i]);
        ensure_post_init();
        expect(legacy, 1);
    }

    reset("1");
    ensure_post_init();
    expect(broker, 1);
    ensure_post_init();
    expect(broker, 1);  // pthread_once must not discover or size twice.

    reset("1");
    broker_result = NVML_ERROR_NOT_FOUND;
    ensure_post_init();
    expect(broker, 0);  // NVML cannot repair a missing shared-region slot.

    reset("1");
    broker_result = NVML_ERROR_UNKNOWN;
    ensure_post_init();
    expect(fallback, 1);

    reset("1");
    broker_result = NVML_ERROR_UNKNOWN;
    probe_result = NVML_ERROR_DRIVER_NOT_LOADED;
    ensure_post_init();
    expect(fallback, 0);

    reset("1");
    broker_result = NVML_ERROR_UNKNOWN;
    deadline_result = -1;
    ensure_post_init();
    expect("allocator map broker deadline utilization watcher ", 0);

    static const int errors[] = {EACCES, ENOENT, ETIMEDOUT};
    for (size_t i = 0; i < sizeof(errors) / sizeof(errors[0]); i++) {
        reset("1");
        broker_result = NVML_ERROR_UNKNOWN;
        node_lock_errno = errors[i];
        ensure_post_init();
        expect("allocator map broker deadline node_lock utilization watcher ", 0);
    }

    reset("1");
    broker_result = NVML_ERROR_UNKNOWN;
    cache_lock_result = 0;
    ensure_post_init();
    expect("allocator map broker deadline node_lock cache_lock "
           "node_unlock utilization watcher ", 0);

    reset("1");
    broker_result = NVML_ERROR_UNKNOWN;
    release_result = -1;
    ensure_post_init();
    expect(fallback, 1);

    reset(NULL);
    cache_lock_result = 0;
    ensure_post_init();
    expect("allocator map cache_lock utilization watcher ", 0);

    reset(NULL);
    probe_result = NVML_ERROR_UNKNOWN;
    ensure_post_init();
    expect(legacy, 0);

    reset("1");
    ensure_post_init();
    events[0] = '\0';
    node_locked = 1;  // The child may inherit an in-flight fallback lock.
    childReinitPostInit();
    check(strcmp(events, "fork_reset ") == 0, "child discards inherited lock");
    check(pidfound == 0, "child must rediscover its own PID");
    events[0] = '\0';
    ensure_post_init();
    expect(broker, 1);

    if (failures != 0) {
        fprintf(stderr, "%d host PID broker init checks failed\n", failures);
        return 1;
    }
    puts("host PID broker init tests passed");
    return 0;
}

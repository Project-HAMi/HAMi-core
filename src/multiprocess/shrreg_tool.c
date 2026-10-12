#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <limits.h>
#include <inttypes.h>
#include <fcntl.h>
#include <errno.h>
#include <stdatomic.h>

#include "include/memory_limit.h"


void create_new() {
    load_env_from_file(ENV_OVERRIDE_FILE);
    umask(000);
	char* shrreg_file = getenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV);
    if (shrreg_file == NULL) {
        shrreg_file = MULTIPROCESS_SHARED_REGION_CACHE_DEFAULT;
    }
    int fd = open(shrreg_file, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
    	LOG_ERROR("Fail to create new shrreg file\n");
        assert(0);
    }
    close(fd);
    ensure_initialized();
}


void print_shared_region(){
    ensure_initialized();
    print_all();
}


void send_stop_signal(){
    ensure_initialized();
    suspend_all();
    while (1){
        if (wait_status_all(2))
            break;
        sleep(1);
    }
}

void send_resume_signal(){
    ensure_initialized();
    resume_all();
    while (1){
        if (wait_status_all(1))
            break;
        sleep(1);
    }
}

typedef struct {
    const char *code;
    const char *message;
} diagnostic_finding_t;

typedef struct {
    diagnostic_finding_t findings[8];
    size_t finding_count;
    int initialized;
    uint32_t major_version;
    uint32_t minor_version;
    uint64_t device_num;
    uint64_t owner_pid;
    int proc_num;
    int occupied_slots;
    int empty_slots;
    int unstable_slots;
    uint64_t limits[CUDA_DEVICE_MAX_COUNT];
    uint64_t sm_limits[CUDA_DEVICE_MAX_COUNT];
    int data_available;
} diagnostic_report_t;

static void add_finding(diagnostic_report_t *report, const char *code,
                        const char *message) {
    if (report->finding_count < sizeof(report->findings) / sizeof(report->findings[0])) {
        report->findings[report->finding_count].code = code;
        report->findings[report->finding_count].message = message;
        report->finding_count++;
    }
}

static int resolve_cache_path(char *path, size_t path_size) {
    const char *configured = getenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV);
    if (configured != NULL) {
        int n = snprintf(path, path_size, "%s", configured);
        return n >= 0 && (size_t)n < path_size;
    }

    const char *hook_path = getenv(MULTIPROCESS_SHARED_REGION_HOOK_PATH_ENV);
    const char *pod_uid = getenv(MULTIPROCESS_SHARED_REGION_POD_UID_ENV);
    const char *container_name = getenv(MULTIPROCESS_SHARED_REGION_CONTAINER_NAME_ENV);
    if (hook_path == NULL || hook_path[0] == '\0')
        hook_path = MULTIPROCESS_SHARED_REGION_HOOK_PATH_DEFAULT;
    if (pod_uid != NULL && pod_uid[0] != '\0' &&
        container_name != NULL && container_name[0] != '\0') {
        int n = snprintf(path, path_size, "%s/containers/%s_%s/usage.cache",
                         hook_path, pod_uid, container_name);
        return n >= 0 && (size_t)n < path_size;
    }
    int n = snprintf(path, path_size, "%s", MULTIPROCESS_SHARED_REGION_CACHE_DEFAULT);
    return n >= 0 && (size_t)n < path_size;
}

static void inspect_shared_region(const char *path, diagnostic_report_t *report) {
    memset(report, 0, sizeof(*report));
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        add_finding(report, errno == ENOENT ? "missing_file" : "open_failed",
                    errno == ENOENT ? "shared-region cache does not exist" :
                                      "shared-region cache could not be opened");
        return;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        add_finding(report, "stat_failed", "could not inspect shared-region cache metadata");
        close(fd);
        return;
    }
    if (st.st_size < (off_t)sizeof(shared_region_t)) {
        add_finding(report, "truncated_file", "shared-region cache is smaller than the expected structure");
        close(fd);
        return;
    }

    /* Coordinate with initialization/truncation performed under the existing file lock. */
    struct flock lock = {0};
    lock.l_type = F_RDLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = (off_t)sizeof(shared_region_t);
    while (fcntl(fd, F_SETLKW, &lock) != 0) {
        if (errno == EINTR)
            continue;
        add_finding(report, "lock_failed", "could not acquire a read lock on the cache");
        close(fd);
        return;
    }

    /* Recheck under the lock before mapping; a short file must never be mapped. */
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(shared_region_t)) {
        add_finding(report, "truncated_file", "shared-region cache became too small before inspection");
        lock.l_type = F_UNLCK;
        (void)fcntl(fd, F_SETLK, &lock);
        close(fd);
        return;
    }

    shared_region_t *region = mmap(NULL, sizeof(*region), PROT_READ, MAP_SHARED, fd, 0);
    if (region == MAP_FAILED) {
        add_finding(report, "map_failed", "could not map the shared-region cache read-only");
        lock.l_type = F_UNLCK;
        (void)fcntl(fd, F_SETLK, &lock);
        close(fd);
        return;
    }

    report->data_available = 1;
    int init_flag = atomic_load_explicit(&region->initialized_flag, memory_order_acquire);
    report->initialized = init_flag == MULTIPROCESS_SHARED_REGION_MAGIC_FLAG;
    report->major_version = region->major_version;
    report->minor_version = region->minor_version;
    if (!report->initialized) {
        add_finding(report, "uninitialized", "shared-region initialization flag is not valid");
    } else {
        if (report->major_version != MAJOR_VERSION || report->minor_version != MINOR_VERSION)
            add_finding(report, "incompatible_version", "shared-region version does not match this tool");

        report->device_num = region->device_num;
        report->owner_pid = atomic_load_explicit(&region->owner_pid, memory_order_acquire);
        report->proc_num = atomic_load_explicit(&region->proc_num, memory_order_acquire);
        if (report->device_num > CUDA_DEVICE_MAX_COUNT)
            add_finding(report, "device_count_out_of_range", "device count exceeds the supported maximum");
        if (report->proc_num < 0 || report->proc_num > SHARED_REGION_MAX_PROCESS_NUM) {
            add_finding(report, "process_count_out_of_range", "process count is outside the supported range");
        } else {
            int slot_snapshot_unstable = 0;
            for (int i = 0; i < report->proc_num; ++i) {
                shrreg_proc_slot_t *slot = &region->procs[i];
                uint64_t before = atomic_load_explicit(&slot->seqlock, memory_order_acquire);
                int32_t pid_before = atomic_load_explicit(&slot->pid, memory_order_acquire);
                int32_t pid_after = atomic_load_explicit(&slot->pid, memory_order_acquire);
                uint64_t after = atomic_load_explicit(&slot->seqlock, memory_order_acquire);
                if ((before & 1U) != 0 || before != after || pid_before != pid_after) {
                    report->unstable_slots++;
                    slot_snapshot_unstable = 1;
                    continue;
                }
                if (pid_before == 0)
                    report->empty_slots++;
                else
                    report->occupied_slots++;
            }

            int final_proc_num = atomic_load_explicit(&region->proc_num, memory_order_acquire);
            if (final_proc_num != report->proc_num)
                slot_snapshot_unstable = 1;
            if (slot_snapshot_unstable)
                add_finding(report, "concurrent_slot_update", "one or more process slots changed during inspection");
            if (report->proc_num == SHARED_REGION_MAX_PROCESS_NUM)
                add_finding(report, "process_slots_exhausted", "the process-slot table is at capacity");
        }
        size_t devices_to_read = report->device_num < CUDA_DEVICE_MAX_COUNT ?
                                 (size_t)report->device_num : CUDA_DEVICE_MAX_COUNT;
        for (size_t i = 0; i < devices_to_read; ++i) {
            report->limits[i] = region->limit[i];
            report->sm_limits[i] = region->sm_limit[i];
        }
    }

    (void)munmap(region, sizeof(*region));
    lock.l_type = F_UNLCK;
    (void)fcntl(fd, F_SETLK, &lock);
    close(fd);
}

static void print_json_string(const char *value) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        if (*p == '"' || *p == '\\') {
            putchar('\\');
            putchar(*p);
        } else if (*p < 0x20 || *p >= 0x80) {
            printf("\\u%04x", *p);
        } else {
            putchar(*p);
        }
    }
    putchar('"');
}

static void print_diagnostic_json(const char *path, const diagnostic_report_t *report) {
    printf("{\"path\":"); print_json_string(path);
    printf(",\"initialized\":%s,\"version\":{\"major\":%u,\"minor\":%u}",
           report->initialized ? "true" : "false", report->major_version, report->minor_version);
    if (report->data_available) {
        printf(",\"device_count\":%" PRIu64 ",\"devices\":[", report->device_num);
        size_t count = report->device_num < CUDA_DEVICE_MAX_COUNT ?
                       (size_t)report->device_num : CUDA_DEVICE_MAX_COUNT;
        for (size_t i = 0; i < count; ++i) {
            if (i) putchar(',');
            printf("{\"index\":%zu,\"memory_limit\":%" PRIu64 ",\"sm_limit\":%" PRIu64 "}",
                   i, report->limits[i], report->sm_limits[i]);
        }
        printf("],\"process_slots\":{\"declared\":%d,\"occupied\":%d,"
               "\"empty\":%d,\"unstable\":%d},"
               "\"owner_pid\":%" PRIu64 ",\"owner_status\":\"%s\"",
               report->proc_num, report->occupied_slots, report->empty_slots,
               report->unstable_slots, report->owner_pid,
               report->owner_pid == 0 ? "unowned" : "pid_recorded_liveness_unknown");
    }
    printf(",\"findings\":[");
    for (size_t i = 0; i < report->finding_count; ++i) {
        if (i) putchar(',');
        printf("{\"code\":"); print_json_string(report->findings[i].code);
        printf(",\"message\":"); print_json_string(report->findings[i].message);
        putchar('}');
    }
    printf("]}\n");
}

static void print_diagnostic_text(const char *path, const diagnostic_report_t *report) {
    printf("Shared-region cache: %s\n", path);
    printf("Initialized: %s\n", report->initialized ? "yes" : "no");
    if (report->data_available) {
        printf("Version: %u.%u\n", report->major_version, report->minor_version);
        printf("Devices: %" PRIu64 "\n", report->device_num);
        size_t count = report->device_num < CUDA_DEVICE_MAX_COUNT ?
                       (size_t)report->device_num : CUDA_DEVICE_MAX_COUNT;
        for (size_t i = 0; i < count; ++i)
            printf("Device %zu limits: memory=%" PRIu64 " bytes, SM=%" PRIu64 "%%\n",
                   i, report->limits[i], report->sm_limits[i]);
        printf("Process slots: declared=%d occupied=%d empty=%d unstable=%d\n",
               report->proc_num, report->occupied_slots, report->empty_slots,
               report->unstable_slots);
        printf("Owner PID: %" PRIu64 " (%s)\n", report->owner_pid,
               report->owner_pid == 0 ? "unowned" : "PID recorded; liveness unknown");
    }
    for (size_t i = 0; i < report->finding_count; ++i)
        printf("Finding [%s]: %s\n", report->findings[i].code,
               report->findings[i].message);
    if (report->finding_count == 0)
        puts("No validation findings.");
}

static int run_diagnostics(int json, int check) {
    char path[PATH_MAX];
    /* Honor the same override file as normal initialization without opening the cache for writing. */
    (void)load_env_from_file(ENV_OVERRIDE_FILE);
    if (!resolve_cache_path(path, sizeof(path))) {
        fputs("shared-region cache path is too long\n", stderr);
        return 2;
    }
    diagnostic_report_t report;
    inspect_shared_region(path, &report);
    if (json)
        print_diagnostic_json(path, &report);
    else
        print_diagnostic_text(path, &report);
    return check && report.finding_count != 0 ? 1 : 0;
}

int main(int argc, char* argv[]) {
	int k;
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        printf(
            "--create_new    Create new shared region file\n"
            "--check         Read-only validate the existing shared region\n"
            "--json          Emit a JSON shared-region diagnostic report\n"
        );
        return 0;
    }

    int check = 0, json = 0;
    for (k = 1; k < argc; ++k) {
        if (strcmp(argv[k], "--check") == 0) check = 1;
        if (strcmp(argv[k], "--json") == 0) json = 1;
    }
    if (check || json)
        return run_diagnostics(json, check);

    for (k = 0; k < argc; ++k) {
        char* arg = argv[k];
        if (strcmp(arg, "--create_new") == 0) {
            create_new();
        }
        if (strcmp(arg, "--suspend") == 0){
            send_stop_signal();
        }
        if (strcmp(arg, "--print") == 0){
            print_shared_region();
        }
        if (strcmp(arg, "--resume") == 0){
            send_resume_signal();
        }
        if (strcmp(arg, "--version") == 0){
            printf("shrreg size: %ld, version %d.%d\n", 
                    sizeof(shared_region_t),
                    shrreg_major_version(),
                    shrreg_minor_version());
        }
    }
    return 0;
}


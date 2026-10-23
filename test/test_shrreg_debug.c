#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        dprintf(saved_stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static off_t log_size(FILE *log) {
    struct stat statbuf;
    return fstat(fileno(log), &statbuf) == 0 ? statbuf.st_size : -1;
}

int main(void) {
    char path[] = "/tmp/hami-shrreg-debug.XXXXXX";
    int fd = mkstemp(path);
    int saved_stderr = dup(STDERR_FILENO);
    FILE *log = tmpfile();
    REQUIRE(fd >= 0 && saved_stderr >= 0 && log != NULL);
    close(fd);
    unlink(path);
    REQUIRE(setenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV, path, 1) == 0);
    REQUIRE(setenv("LIBCUDA_LOG_LEVEL", "4", 1) == 0);
    REQUIRE(unlink(ENV_OVERRIDE_FILE) == 0 || errno == ENOENT);
    REQUIRE(dup2(fileno(log), STDERR_FILENO) >= 0);
    log_utils_init();
    ensure_initialized();

    fd = open(path, O_RDWR);
    REQUIRE(fd >= 0);
    shared_region_t *region = mmap(NULL, sizeof(*region), PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd, 0);
    REQUIRE(region != MAP_FAILED);
    REQUIRE(atomic_load(&region->proc_num) == 1);
    region->device_num = 3;
    shrreg_proc_slot_t *slot = &region->procs[0];
    REQUIRE(atomic_load(&slot->pid) == getpid());
    atomic_store(&slot->hostpid, 12345);
    atomic_store(&slot->device_util[2].sm_util, 42);
    atomic_store(&slot->monitorused[2], 1000);
    atomic_store(&slot->used[2].total, 2000);
    atomic_store(&slot->used[3].total, 3000);
    print_all();
    REQUIRE(atomic_load(&slot->hostpid) == 12345);
    REQUIRE(atomic_load(&slot->device_util[2].sm_util) == 42);
    REQUIRE(atomic_load(&slot->monitorused[2]) == 1000);
    REQUIRE(atomic_load(&slot->used[2].total) == 2000);

    fflush(stderr);
    off_t size = log_size(log);
    REQUIRE(size > 0);
    char *output = calloc((size_t)size + 1, 1);
    REQUIRE(output != NULL);
    rewind(log);
    REQUIRE(fread(output, 1, (size_t)size, log) == (size_t)size);
    const char *first = strstr(output, "Shared-region snapshot: total processes: 1");
    REQUIRE(first != NULL);
    REQUIRE(strstr(first + 1, "Shared-region snapshot: total processes: 1") == NULL);
    char expected[256];
    snprintf(expected, sizeof(expected),
             "Shared-region process %d hostPid: 12345 device: 2, sm: 42, memory: 1000, record: 2000",
             (int)getpid());
    REQUIRE(strstr(output, expected) != NULL);
    REQUIRE(strstr(output, "device: 0,") == NULL);
    REQUIRE(strstr(output, "device: 3,") == NULL);
    free(output);

    region->device_num = UINT64_MAX;
    atomic_store(&slot->used[15].total, 15000);
    print_all();
    region->device_num = 3;

    for (int i = 1; i < 20; i++) {
        atomic_store(&region->procs[i].pid, 20000 + i);
    }
    atomic_store(&slot->pid, 20000);
    atomic_store(&region->procs[18].pid, getpid());
    atomic_store(&region->proc_num, 20);
    print_all();
    atomic_store(&region->proc_num, 1);
    atomic_store(&slot->pid, getpid());

    REQUIRE(fseek(log, 0, SEEK_END) == 0);
    size = log_size(log);
    g_log_level = 3;
    print_all();
    fflush(stderr);
    REQUIRE(log_size(log) == size);
    g_log_level = 4;
    REQUIRE(sem_wait(&region->sem) == 0);
    print_all();
    REQUIRE(sem_post(&region->sem) == 0);
    fflush(stderr);
    REQUIRE(log_size(log) > size);
    size = log_size(log);
    atomic_store(&region->proc_num, SHARED_REGION_MAX_PROCESS_NUM + 1);
    print_all();
    atomic_store(&region->proc_num, 1);
    fflush(stderr);
    REQUIRE(log_size(log) > size);
    off_t final_size = log_size(log);
    output = calloc((size_t)final_size + 1, 1);
    REQUIRE(output != NULL);
    rewind(log);
    REQUIRE(fread(output, 1, (size_t)final_size, log) == (size_t)final_size);
    REQUIRE(strstr(output, "Shared-region snapshot skipped: lock unavailable") != NULL);
    REQUIRE(strstr(output, "Invalid shared-region process count: 1025") != NULL);
    REQUIRE(strstr(output, "device: 15, sm: 0, memory: 0, record: 15000") != NULL);
    const char *bounded = strstr(output, "total processes: 20, shown: 16");
    REQUIRE(bounded != NULL);
    snprintf(expected, sizeof(expected), "Shared-region process %d hostPid: 0", (int)getpid());
    REQUIRE(strstr(bounded, expected) != NULL);
    REQUIRE(strstr(output, "Shared-region process 20016 hostPid:") == NULL);
    REQUIRE(strstr(output, "Shared-region process 20019 hostPid:") == NULL);
    free(output);

    munmap(region, sizeof(*region));
    close(fd);
    unlink(path);
    REQUIRE(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);
    fclose(log);
    puts("shared-region debug snapshot tests passed");
    return 0;
}

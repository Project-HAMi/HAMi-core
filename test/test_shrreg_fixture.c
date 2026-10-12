#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#include "multiprocess/multiprocess_memory_limit.h"

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *file = fopen(argv[1], "wb");
    if (file == NULL) return 3;
    if (strcmp(argv[2], "truncated") == 0) {
        char bytes[32] = {0};
        int result = fwrite(bytes, sizeof(bytes), 1, file) == 1 ? 0 : 4;
        fclose(file);
        return result;
    }

    shared_region_t region;
    memset(&region, 0, sizeof(region));
    if (strcmp(argv[2], "uninitialized") != 0)
        atomic_store_explicit(&region.initialized_flag,
                              MULTIPROCESS_SHARED_REGION_MAGIC_FLAG,
                              memory_order_relaxed);
    region.major_version = MAJOR_VERSION;
    region.minor_version = MINOR_VERSION;
    region.device_num = 2;
    if (strcmp(argv[2], "bad_device_count") == 0)
        region.device_num = CUDA_DEVICE_MAX_COUNT + 1;
    region.limit[0] = 1024;
    region.limit[1] = 2048;
    region.sm_limit[0] = 50;
    region.sm_limit[1] = 75;
    atomic_store_explicit(&region.owner_pid,
                          strcmp(argv[2], "owned") == 0 ? 4242 : 0,
                          memory_order_relaxed);
    int proc_num = 2;
    if (strcmp(argv[2], "bad_process_count") == 0)
        proc_num = SHARED_REGION_MAX_PROCESS_NUM + 1;
    if (strcmp(argv[2], "exhausted") == 0)
        proc_num = SHARED_REGION_MAX_PROCESS_NUM;
    atomic_store_explicit(&region.proc_num, proc_num, memory_order_relaxed);
    if (proc_num <= SHARED_REGION_MAX_PROCESS_NUM) {
        for (int i = 0; i < proc_num; ++i) {
            atomic_store_explicit(&region.procs[i].pid, i == 1 ? 0 : 1234 + i,
                                  memory_order_relaxed);
            atomic_store_explicit(&region.procs[i].seqlock,
                                  strcmp(argv[2], "unstable_slot") == 0 && i == 0 ? 1 : 0,
                                  memory_order_relaxed);
        }
    }
    if (strcmp(argv[2], "bad_version") == 0)
        region.major_version++;

    int result = fwrite(&region, sizeof(region), 1, file) == 1 ? 0 : 4;
    if (fclose(file) != 0) return 5;
    return result;
}

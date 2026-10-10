#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>



typedef struct nvmlProcessInfo_st1
{
    unsigned int        pid;
    unsigned long long  usedGpuMemory;
} nvmlProcessInfo_t1;

int mergepid(unsigned int *prev, unsigned int *current, nvmlProcessInfo_t1 *sub, nvmlProcessInfo_t1 *merged);
int getextrapid(unsigned int prev, unsigned int current, nvmlProcessInfo_t1 *pre_pids_on_device, nvmlProcessInfo_t1 *pids_on_device);
unsigned int own_pid_candidates(const nvmlProcessInfo_t1 *before, unsigned int n_before,
                                const nvmlProcessInfo_t1 *during, unsigned int n_during,
                                const nvmlProcessInfo_t1 *after, unsigned int n_after,
                                const unsigned int *known, unsigned int n_known,
                                unsigned int *out, unsigned int max_out);
unsigned int hostpid_retry_seed(const struct timespec *ts, unsigned int pid);

#define HOSTPID_PROBE_MAX 1024  // SHARED_REGION_MAX_PROCESS_NUM
typedef struct {
    nvmlProcessInfo_t1 before[HOSTPID_PROBE_MAX], during[HOSTPID_PROBE_MAX], after[HOSTPID_PROBE_MAX];
    unsigned int n_before, n_during, n_after;
} hostpid_probe_t;
typedef int (*hostpid_probe_fn)(void *ctx, hostpid_probe_t *p);
int find_own_hostpid(hostpid_probe_fn probe, void *ctx, int max_attempts,
                     unsigned int *hostpid, unsigned long long *used);

//Nvml part utils
void sort(int vmap[16]);
int initial_virtual_devices();
int parser(char *str);
int need_cuda_virtualize();

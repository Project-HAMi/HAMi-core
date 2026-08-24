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

//Nvml part utils
void sort(int vmap[16]);
int initial_virtual_devices();
int parser(char *str);
int need_cuda_virtualize();

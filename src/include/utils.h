#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>


//Nvml part utils
void sort(int vmap[16]);
int initial_virtual_devices();
int parser(char *str);
int need_cuda_virtualize();

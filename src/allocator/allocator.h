#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <memory.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>

struct allocated_device_memory_struct{
    CUdeviceptr address;
    size_t length;
    CUdevice dev;
};
typedef struct allocated_device_memory_struct allocated_device_memory;

struct allocated_list_entry_struct{
    allocated_device_memory *entry;
    struct allocated_list_entry_struct *next,*prev;
};
typedef struct allocated_list_entry_struct allocated_list_entry;

struct allocated_list_struct{
    allocated_list_entry *head;
    allocated_list_entry *tail;
    size_t length;
    size_t limit;
};
typedef struct allocated_list_struct allocated_list;

extern allocated_list *device_overallocated;
extern allocated_list *device_allocasync;
extern pthread_mutex_t mutex;

#define LIST_INIT(list) {   \
    list->head=NULL;         \
    list->tail=NULL;         \
    list->length=0;          \
    list->limit=0;           \
    }

#define QUIT_WITH_ERROR(__message) {    \
    LOG_ERROR("%s\n",#__message);  \
    return -1;                          \
}

#define LIST_REMOVE(list,val) {             \
    if (val->prev!=NULL)                    \
        val->prev->next=val->next;          \
    if (val->next!=NULL)                    \
        val->next->prev=val->prev;          \
    if (val == list->tail)                  \
        list->tail = val->prev;             \
    if (val == list->head)                  \
        list->head = val->next;             \
    free(val->entry);                       \
    free(val);                              \
    list->length--;                         \
}   

#define INIT_ALLOCATED_LIST_ENTRY(__list_entry, __address, __size, __dev) {             \
    __list_entry = malloc(sizeof(allocated_list_entry));                       \
    if (__list_entry == NULL) QUIT_WITH_ERROR("malloc failed");                \
    __list_entry->entry = malloc(sizeof(allocated_device_memory));             \
    if (__list_entry->entry == NULL) QUIT_WITH_ERROR("malloc failed");         \
    __list_entry->entry->address=__address;                                    \
    __list_entry->entry->length=__size;                                        \
    __list_entry->entry->dev = __dev;                                            \
    __list_entry->next=NULL;                                                   \
    __list_entry->prev=NULL;                                                   \
}

#define LIST_ADD(list,__entry) { \
    if (list->head == NULL) {    \
        list->head = __entry;    \
        list->tail = __entry;    \
        list->length=1;          \
    }else{                      \
        __entry->prev = list->tail; \
        list->tail->next=__entry;   \
        list->tail = __entry;       \
        list->length++;             \
    }                               \
}                                   


// Checks if oom
int oom_check(const int dev,size_t addon);

/* Cross-process check-and-reserve under lock_shrreg.
 * Returns 0 on success, CUDA_ERROR_OUT_OF_MEMORY if the limit would be exceeded.
 * On success, size is already accounted in shared usage; call
 * release_device_memory() if the subsequent CUDA alloc fails. */
int reserve_device_memory(CUdevice dev, size_t size);
void release_device_memory(CUdevice dev, size_t size);

// Allocate and free device memory
int allocate_raw(CUdeviceptr *dptr, size_t size);
int free_raw(CUdeviceptr dptr);
int add_chunk_only(CUdeviceptr address, size_t size, CUdevice dev);
int remove_chunk_only(CUdeviceptr address);
int allocate_async_raw(CUdeviceptr *dptr, size_t size, CUstream hStream);
int allocate_from_pool_async_raw(CUdeviceptr *dptr, size_t size, CUmemoryPool pool, CUstream hStream);
int free_raw_async(CUdeviceptr dptr, CUstream hStream);

// Checks memory type
int check_memory_type(CUdeviceptr address);


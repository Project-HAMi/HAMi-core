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

/* Reserve graph-allocation quota and roll it back if the native call fails. */
typedef struct graph_charge_entry graph_charge_entry;
int graph_alloc_begin(CUgraph graph, CUdevice dev, size_t size, cuuint64_t capid, int capturing,
                      graph_charge_entry **out);
void graph_alloc_end(graph_charge_entry *entry, CUdeviceptr ptr, CUresult result);
void graph_mark_free(CUgraph graph, CUdeviceptr ptr);
void graph_ptr_freed(CUdeviceptr ptr);
void graph_ptr_freed_async(CUdeviceptr ptr, CUstream hStream);
void graph_release(CUgraph graph);
typedef struct graph_exec_state graph_exec_state;
int graph_exec_prepare(CUgraph graph, graph_exec_state **out);
void graph_exec_commit(graph_exec_state *state, CUgraphExec exec);
void graph_exec_discard(graph_exec_state *state);
void graph_exec_destroyed(CUgraphExec exec);
int graph_exec_reserve(graph_exec_state *state, CUstream stream);
int graph_exec_begin(CUgraphExec exec, CUstream stream, graph_exec_state **out);
void graph_exec_end(graph_exec_state *state, CUstream stream, CUresult result);
cuuint64_t graph_context_id(CUcontext ctx);
void graph_context_destroyed(CUcontext ctx, cuuint64_t id);
void graph_primary_context(CUdevice dev, CUcontext ctx);
void graph_primary_context_released(CUdevice dev, int reset);
void graph_stream_destroyed(CUstream stream);
uint64_t graph_sync_begin(void);
void graph_synchronized(CUcontext ctx, CUstream stream, int whole_context,
                        uint64_t submitted);
void graph_pool_trimmed(void);
CUresult graph_capture_info(CUstream hStream, CUstreamCaptureStatus *cap, cuuint64_t *id, CUgraph *graph);
int graph_capture_prepare(CUstream stream, CUstreamCaptureMode mode, uint64_t *token);
void graph_capture_begin(uint64_t token, CUresult result);
uint64_t graph_capture_token(CUstream stream, cuuint64_t *id);
void graph_capture_finish(uint64_t token);
cuuint64_t graph_capture_id_of(CUgraph graph);
void graph_capture_end(cuuint64_t capid, CUgraph graph);

// Allocate and free device memory
int allocate_raw(CUdeviceptr *dptr, size_t size);
int free_raw(CUdeviceptr dptr);
int add_chunk_only(CUdeviceptr address, size_t size, CUdevice dev);
int remove_chunk_only(CUdeviceptr address);

/* VMM accounting: the charge from cuMemCreate is dropped only after the
 * handle is fully released and all of its mappings are unmapped. */
int vmm_handle_created(CUmemGenericAllocationHandle handle);
void vmm_handle_retained(CUmemGenericAllocationHandle handle);
void vmm_handle_released(CUmemGenericAllocationHandle handle);
typedef struct vmm_map_entry vmm_map_entry;
int vmm_map_begin(CUmemGenericAllocationHandle handle, vmm_map_entry **out);
void vmm_map_end(vmm_map_entry *m, CUdeviceptr ptr, int mapped);
void vmm_unmapped(CUdeviceptr ptr, size_t size);
int allocate_async_raw(CUdeviceptr *dptr, size_t size, CUstream hStream);
int allocate_from_pool_async_raw(CUdeviceptr *dptr, size_t size, CUmemoryPool pool, CUstream hStream);
int free_raw_async(CUdeviceptr dptr, CUstream hStream);

// Checks memory type
int check_memory_type(CUdeviceptr address);


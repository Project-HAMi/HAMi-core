#include "allocator.h"
#include "include/log_utils.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"


allocated_list *device_overallocated;
allocated_list *device_allocasync;

extern CUresult
    cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, void* data);
extern CUresult cuMemoryFree(CUdeviceptr dptr);

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/* already_locked: caller already holds lock_shrreg (not reentrant). */
static int oom_check_impl(const int dev, size_t addon, int already_locked) {
    CUdevice d;
    if (dev == -1) {
        if (cuCtxGetDevice(&d) != CUDA_SUCCESS) {
            LOG_WARN("oom_check: no current context, skipping enforcement");
            return 0;
        }
    } else {
        d=dev;
    }
    uint64_t limit = get_current_device_memory_limit(d);
    /* limit[] is CUDA indexed, used[] is NVML indexed. */
    size_t _usage = get_gpu_memory_usage(cuda_to_nvml_map(d));

    if (limit == 0) {
        return 0;
    }

    size_t new_allocated = _usage + addon;
    LOG_INFO("_usage=%lu limit=%lu new_allocated=%lu",_usage,limit,new_allocated);
    if (new_allocated > limit) {
        int cleared;

        LOG_ERROR("Device %d OOM %lu / %lu", d, new_allocated, limit);

        if (already_locked) {
            cleared = clear_proc_slot_nolock(1);
        } else {
            lock_shrreg();
            cleared = clear_proc_slot_nolock(1);
            unlock_shrreg();
        }
        if (cleared > 0)
            return oom_check_impl(dev, addon, already_locked);
        return 1;
    }
    return 0;
}

int oom_check(const int dev, size_t addon) {
    return oom_check_impl(dev, addon, 0);
}

int reserve_device_memory(CUdevice dev, size_t size) {
    lock_shrreg();
    if (oom_check_impl(dev, size, 1)) {
        unlock_shrreg();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    add_gpu_device_memory_usage(getpid(), dev, size, 2);
    unlock_shrreg();
    return 0;
}

void release_device_memory(CUdevice dev, size_t size) {
    lock_shrreg();
    rm_gpu_device_memory_usage(getpid(), dev, size, 2);
    unlock_shrreg();
}

void allocator_init() {
    LOG_DEBUG("Allocator_init\n");

    device_overallocated = malloc(sizeof(allocated_list));
    if (!device_overallocated) {
        LOG_ERROR("allocator_init: malloc failed");
        exit(EXIT_FAILURE);
    }
    LIST_INIT(device_overallocated);
    device_allocasync = malloc(sizeof(allocated_list));
    if (!device_allocasync) {
        LOG_ERROR("allocator_init: malloc failed");
        exit(EXIT_FAILURE);
    }
    LIST_INIT(device_allocasync);

    pthread_mutex_init(&mutex,NULL);
}

/* Wrap INIT_ALLOCATED_LIST_ENTRY so QUIT_WITH_ERROR returns here, not from
 * callers that hold mutex / CUDA memory / shared reservations. */
static int new_allocated_list_entry(allocated_list_entry **out,
                                    CUdeviceptr address, size_t size,
                                    CUdevice dev) {
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e, address, size, dev);
    *out = e;
    return 0;
}

int add_chunk(CUdeviceptr *address, size_t size) {
    CUdevice dev;
    CUresult res;
    allocated_list_entry *e;

    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) {
        /* No current context on this thread (for example a worker thread that
         * never bound one). We cannot attribute or bound this allocation to a
         * device, so forward it to the driver rather than index per-device
         * state with an undefined id. */
        LOG_WARN("add_chunk: no current context, forwarding allocation without tracking");
        return cuMemoryAllocate(address, size, NULL);
    }

    /* Reserve under the shared-region lock so concurrent processes cannot
     * both pass oom_check before either commits usage. CUDA alloc stays
     * outside the lock. */
    if (reserve_device_memory(dev, size) != 0)
        return CUDA_ERROR_OUT_OF_MEMORY;

    /* GPU allocation outside lock, the expensive part */
    res = cuMemoryAllocate(address, size, NULL);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemoryAllocate failed res=%d", res);
        release_device_memory(dev, size);
        return res;
    }

    /* Local list tracking only — usage already reserved */
    pthread_mutex_lock(&mutex);
    if (new_allocated_list_entry(&e, 0, size, dev) != 0) {
        pthread_mutex_unlock(&mutex);
        cuMemoryFree(*address);
        release_device_memory(dev, size);
        return -1;
    }
    e->entry->address = *address;
    LIST_ADD(device_overallocated, e);
    pthread_mutex_unlock(&mutex);
    return 0;
}

/* Track a pointer in the local list. Caller must already have reserved
 * `size` via reserve_device_memory() (or equivalent usage accounting). */
int add_chunk_only(CUdeviceptr address, size_t size, CUdevice dev) {
    allocated_list_entry *e;

    pthread_mutex_lock(&mutex);
    if (new_allocated_list_entry(&e, 0, size, dev) != 0) {
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    e->entry->address = address;
    LIST_ADD(device_overallocated, e);
    pthread_mutex_unlock(&mutex);
    return 0;
}

int check_memory_type(CUdeviceptr address) {
    allocated_list_entry *cursor;
    for (cursor=device_overallocated->head;cursor!=NULL;cursor=cursor->next){
        if ((cursor->entry->address <= address) && (cursor->entry->address+cursor->entry->length>=address))
            return CU_MEMORYTYPE_DEVICE;
    }
    return CU_MEMORYTYPE_HOST;
}

int remove_chunk(allocated_list *a_list, CUdeviceptr dptr) {
    size_t t_size;
    CUdevice t_dev;

    if (a_list->length == 0) {
        /* Nothing tracked here, so the pointer was not allocated through this
         * list. Forward the free to the real driver rather than returning -1
         * (see the not-found path below). */
        return cuMemoryFree(dptr);
    }

    pthread_mutex_lock(&mutex);

    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size = val->entry->length;
            t_dev = val->entry->dev;
            LIST_REMOVE(a_list, val);
            rm_gpu_device_memory_usage(getpid(), t_dev, t_size, 2);

            pthread_mutex_unlock(&mutex);

            /* GPU free outside lock */
            cuMemoryFree(dptr);
            return 0;
        }
    }

    pthread_mutex_unlock(&mutex);
    /* Not tracked in this list (e.g. a stream-ordered allocation reaching the
     * synchronous free path): forward the free to the real driver instead of
     * returning -1, which leaves the allocation live and surfaces to the caller
     * as an unrecognized CUresult. Mirrors the not-found path in
     * remove_chunk_async. */
    return cuMemoryFree(dptr);
}

int remove_chunk_only(CUdeviceptr dptr) {
    allocated_list *a_list = device_overallocated;
    size_t t_size;
    CUdevice t_dev;
    if (a_list->length == 0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size = val->entry->length;
            t_dev = val->entry->dev;
            LIST_REMOVE(a_list, val);
            rm_gpu_device_memory_usage(getpid(), t_dev, t_size, 2);
            return 0;
        }
    }
    return -1;
}

/* VMM accounting. Memory from cuMemCreate stays allocated in the driver until
 * every handle reference is released and every mapping of it is unmapped, so
 * the charge is held until both are true. */
typedef struct vmm_handle_entry {
    CUmemGenericAllocationHandle handle;
    int refs;   /* create + retain - release */
    int nmaps;  /* live cuMemMap ranges */
    struct vmm_handle_entry *next;
} vmm_handle_entry;
typedef struct vmm_map_entry {
    CUdeviceptr ptr;
    vmm_handle_entry *h;
    struct vmm_map_entry *next;
} vmm_map_entry;
static vmm_handle_entry *vmm_handles;
static vmm_map_entry *vmm_maps;
static pthread_mutex_t vmm_mutex = PTHREAD_MUTEX_INITIALIZER;

static vmm_handle_entry *vmm_find(CUmemGenericAllocationHandle handle) {
    for (vmm_handle_entry *h = vmm_handles; h != NULL; h = h->next) {
        if (h->handle == handle) return h;
    }
    return NULL;
}

/* Drop the charge once nothing keeps the memory alive. Call with vmm_mutex held. */
static void vmm_maybe_drop(vmm_handle_entry *h) {
    if (h->refs > 0 || h->nmaps > 0) return;
    vmm_handle_entry **pp = &vmm_handles;
    while (*pp != h) pp = &(*pp)->next;
    *pp = h->next;
    remove_chunk_only(h->handle);
    free(h);
}

int vmm_handle_created(CUmemGenericAllocationHandle handle) {
    vmm_handle_entry *h = calloc(1, sizeof(*h));
    if (h == NULL) return -1;
    h->handle = handle;
    h->refs = 1;
    pthread_mutex_lock(&vmm_mutex);
    h->next = vmm_handles;
    vmm_handles = h;
    pthread_mutex_unlock(&vmm_mutex);
    return 0;
}

void vmm_handle_retained(CUmemGenericAllocationHandle handle) {
    pthread_mutex_lock(&vmm_mutex);
    vmm_handle_entry *h = vmm_find(handle);
    if (h != NULL) h->refs++;
    pthread_mutex_unlock(&vmm_mutex);
}

void vmm_handle_released(CUmemGenericAllocationHandle handle) {
    pthread_mutex_lock(&vmm_mutex);
    vmm_handle_entry *h = vmm_find(handle);
    if (h == NULL) {
        /* Not tracked here (e.g. bookkeeping failed at create time). */
        remove_chunk_only(handle);
    } else if (h->refs > 0) {
        h->refs--;
        vmm_maybe_drop(h);
    }
    pthread_mutex_unlock(&vmm_mutex);
}

void vmm_mapped(CUdeviceptr ptr, CUmemGenericAllocationHandle handle) {
    pthread_mutex_lock(&vmm_mutex);
    vmm_handle_entry *h = vmm_find(handle);
    if (h != NULL) {
        vmm_map_entry *m = calloc(1, sizeof(*m));
        if (m != NULL) {
            m->ptr = ptr;
            m->h = h;
            m->next = vmm_maps;
            vmm_maps = m;
            h->nmaps++;
        }
    }
    pthread_mutex_unlock(&vmm_mutex);
}

void vmm_unmapped(CUdeviceptr ptr, size_t size) {
    pthread_mutex_lock(&vmm_mutex);
    vmm_map_entry **mp = &vmm_maps;
    while (*mp != NULL) {
        vmm_map_entry *m = *mp;
        if (m->ptr >= ptr && m->ptr - ptr < size) {
            *mp = m->next;
            m->h->nmaps--;
            vmm_maybe_drop(m->h);
            free(m);
        } else {
            mp = &m->next;
        }
    }
    pthread_mutex_unlock(&vmm_mutex);
}

int allocate_raw(CUdeviceptr *dptr, size_t size) {
    return add_chunk(dptr, size);
}

int free_raw(CUdeviceptr dptr) {
    return remove_chunk(device_overallocated, dptr);
}

int remove_chunk_async(
    allocated_list *a_list, CUdeviceptr dptr, CUstream hStream) {
    size_t t_size;
    CUdevice t_dev;
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size=val->entry->length;
            /* Release against the device recorded when the allocation was
             * charged, captured before LIST_REMOVE frees the entry. The
             * freeing thread may have no current context, or a different
             * current device, so asking cuCtxGetDevice here would skip or
             * misattribute the release and leave the usage charged, which
             * later surfaces as a spurious OOM. */
            t_dev = val->entry->dev;
            CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeAsync,dptr,hStream);
            LIST_REMOVE(a_list,val);
            a_list->limit-=t_size;
            rm_gpu_device_memory_usage(getpid(), t_dev, t_size, 2);
            return 0;
        }
    }
    /* Not tracked by libvgpu: free it for real instead of returning -1, which
     * leaked and surfaced as an "unrecognized error code". */
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemFreeAsync, dptr, hStream);
}

int free_raw_async(CUdeviceptr dptr, CUstream hStream) {
    pthread_mutex_lock(&mutex);
    unsigned int tmp = remove_chunk_async(device_allocasync, dptr, hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

typedef struct graph_charge_entry {
    CUgraph graph;       /* NULL once the graph is destroyed */
    CUdevice dev;
    size_t size;
    CUdeviceptr ptr;
    int has_free;
    int nexec;           /* live execs instantiated from the graph */
    struct graph_charge_entry *next;
} graph_charge_entry;
typedef struct graph_exec_link {
    CUgraphExec exec;
    graph_charge_entry *entry;
    struct graph_exec_link *next;
} graph_exec_link;
static graph_charge_entry *graph_charges;
static graph_exec_link *graph_execs;
static pthread_mutex_t graph_mutex = PTHREAD_MUTEX_INITIALIZER;

int graph_record(CUgraph graph, CUdevice dev, size_t size, CUdeviceptr ptr) {
    graph_charge_entry *g = calloc(1, sizeof(*g));
    if (g == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
    g->graph = graph; g->dev = dev; g->size = size; g->ptr = ptr;
    pthread_mutex_lock(&graph_mutex);
    g->next = graph_charges;
    graph_charges = g;
    pthread_mutex_unlock(&graph_mutex);
    return 0;
}

/* Release the charge and forget the entry. Call with graph_mutex held. */
static void graph_drop_entry(graph_charge_entry *e) {
    graph_exec_link **lp = &graph_execs;
    while (*lp != NULL) {
        if ((*lp)->entry == e) {
            graph_exec_link *l = *lp;
            *lp = l->next;
            free(l);
        } else {
            lp = &(*lp)->next;
        }
    }
    graph_charge_entry **pp = &graph_charges;
    while (*pp != e) pp = &(*pp)->next;
    *pp = e->next;
    release_device_memory(e->dev, e->size);
    free(e);
}

/* Only the allocating graph's own free node frees the memory when that graph
 * and its execs are gone. A free node in another graph leaves the charge held
 * until the pointer is freed, which can over-count but never under-count. */
void graph_mark_free(CUgraph graph, CUdeviceptr ptr) {
    pthread_mutex_lock(&graph_mutex);
    for (graph_charge_entry *g = graph_charges; g != NULL; g = g->next) {
        if (g->ptr == ptr && g->graph == graph) g->has_free = 1;
    }
    pthread_mutex_unlock(&graph_mutex);
}

void graph_ptr_freed(CUdeviceptr ptr) {
    pthread_mutex_lock(&graph_mutex);
    for (graph_charge_entry *g = graph_charges, *n; g != NULL; g = n) {
        n = g->next;
        if (g->ptr == ptr) graph_drop_entry(g);
    }
    pthread_mutex_unlock(&graph_mutex);
}

void graph_ptr_freed_async(CUdeviceptr ptr, CUstream hStream) {
    CUstreamCaptureStatus cap = CU_STREAM_CAPTURE_STATUS_NONE;
    cuuint64_t id;
    CUgraph graph = NULL;
    CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamGetCaptureInfo_v2, hStream, &cap, &id,
                       &graph, NULL, NULL);
    if (cap != CU_STREAM_CAPTURE_STATUS_NONE) {
        graph_mark_free(graph, ptr);
    } else {
        graph_ptr_freed(ptr);
    }
}

/* An exec keeps its graph's allocations alive after the graph is destroyed. */
int graph_exec_created(CUgraph graph, CUgraphExec exec) {
    int ret = 0;
    pthread_mutex_lock(&graph_mutex);
    for (graph_charge_entry *g = graph_charges; g != NULL; g = g->next) {
        if (g->graph != graph) continue;
        graph_exec_link *l = malloc(sizeof(*l));
        if (l == NULL) {
            ret = CUDA_ERROR_OUT_OF_MEMORY;
            break;
        }
        l->exec = exec;
        l->entry = g;
        l->next = graph_execs;
        graph_execs = l;
        g->nexec++;
    }
    if (ret != 0) {
        /* Undo the links already made for this exec. */
        graph_exec_link **lp = &graph_execs;
        while (*lp != NULL) {
            graph_exec_link *l = *lp;
            if (l->exec == exec) {
                l->entry->nexec--;
                *lp = l->next;
                free(l);
            } else {
                lp = &l->next;
            }
        }
    }
    pthread_mutex_unlock(&graph_mutex);
    return ret;
}

void graph_exec_destroyed(CUgraphExec exec) {
    pthread_mutex_lock(&graph_mutex);
    graph_exec_link **lp = &graph_execs;
    while (*lp != NULL) {
        graph_exec_link *l = *lp;
        if (l->exec != exec) {
            lp = &l->next;
            continue;
        }
        graph_charge_entry *e = l->entry;
        *lp = l->next;
        free(l);
        if (--e->nexec == 0 && e->graph == NULL && e->has_free) graph_drop_entry(e);
        lp = &graph_execs;  /* graph_drop_entry may have freed other links */
    }
    pthread_mutex_unlock(&graph_mutex);
}

/* Memory with a free node is released once the graph and all its execs are
 * gone. Memory without one stays until its pointer is freed. */
void graph_release(CUgraph graph) {
    pthread_mutex_lock(&graph_mutex);
    for (graph_charge_entry *g = graph_charges, *n; g != NULL; g = n) {
        n = g->next;
        if (g->graph != graph) continue;
        g->graph = NULL;
        if (g->has_free && g->nexec == 0) graph_drop_entry(g);
    }
    pthread_mutex_unlock(&graph_mutex);
}

/* Track an allocated async chunk and account it against its pool's high-water
 * limit. Call with mutex held. */
static int account_async_chunk(allocated_list_entry *e, size_t size, CUmemoryPool pool, CUstream hStream) {
    size_t poollimit;
    /* cuMemPoolGetAttribute is illegal while the stream is capturing and
     * invalidates the capture. The graph owns the memory: charge it to the
     * graph. The pointer is not kept in device_allocasync: its free is passed
     * through untracked, and graph_* releases the charge. */
    CUstreamCaptureStatus cap = CU_STREAM_CAPTURE_STATUS_NONE;
    cuuint64_t capid;
    CUgraph capgraph = NULL;
    CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamGetCaptureInfo_v2, hStream, &cap, &capid,
                       &capgraph, NULL, NULL);
    if (cap != CU_STREAM_CAPTURE_STATUS_NONE) {
        if (capgraph != NULL) {
            int failed = reserve_device_memory(e->entry->dev, size) != 0;
            if (!failed && graph_record(capgraph, e->entry->dev, size, e->entry->address) != 0) {
                release_device_memory(e->entry->dev, size);
                failed = 1;
            }
            if (failed) {
                CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemFreeAsync, e->entry->address, hStream);
                free(e->entry);
                free(e);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }
        free(e->entry);
        free(e);
        return 0;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemPoolGetAttribute, pool,
                                      CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, &poollimit);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemPoolGetAttribute failed res=%d", res);
        return res;
    }
    if (poollimit != 0) {
        if (poollimit > device_allocasync->limit) {
            size_t allocsize = (poollimit - device_allocasync->limit < size)
                               ? poollimit - device_allocasync->limit : size;
            add_gpu_device_memory_usage(getpid(), e->entry->dev, allocsize, 2);
            device_allocasync->limit = device_allocasync->limit + allocsize;
            e->entry->length = allocsize;
        } else {
            e->entry->length = 0;
        }
    } else {
        e->entry->length = 0;
    }
    LIST_ADD(device_allocasync, e);
    return 0;
}

int add_chunk_async(CUdeviceptr *address, size_t size, CUstream hStream) {
    size_t addr = 0;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev, size))
        return CUDA_ERROR_OUT_OF_MEMORY;

    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e, addr, size, dev);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocAsync,&e->entry->address,size,hStream);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemoryAllocate failed res=%d",res);
        return res;
    }
    *address = e->entry->address;
    /* cuMemAllocAsync draws from the device's default pool. */
    CUmemoryPool pool;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetMemPool,&pool,dev);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuDeviceGetMemPool failed res=%d",res);
        return res;
    }
    return account_async_chunk(e, size, pool, hStream);
}

int add_chunk_from_pool_async(CUdeviceptr *address, size_t size, CUmemoryPool pool, CUstream hStream) {
    size_t addr = 0;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev, size))
        return CUDA_ERROR_OUT_OF_MEMORY;

    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e, addr, size, dev);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemAllocFromPoolAsync, &e->entry->address, size, pool, hStream);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemAllocFromPoolAsync failed res=%d", res);
        return res;
    }
    *address = e->entry->address;
    /* Account against the caller's pool, not the default. */
    return account_async_chunk(e, size, pool, hStream);
}

int allocate_async_raw(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk_async(dptr,size,hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int allocate_from_pool_async_raw(CUdeviceptr *dptr, size_t size, CUmemoryPool pool, CUstream hStream) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk_from_pool_async(dptr, size, pool, hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

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

static void graph_reclaim(unsigned int trim_mask);
static unsigned int graph_trim_mask(int dev, size_t addon);

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

/** Retry a refused quota check after releasing unused graph backing. */
int oom_check(const int dev, size_t addon) {
    graph_reclaim(0);
    int ret = oom_check_impl(dev, addon, 0);
    if (ret) {
        unsigned int trim_mask = graph_trim_mask(dev, addon);
        if (trim_mask != 0) graph_reclaim(trim_mask);
        ret = oom_check_impl(dev, addon, 0);
    }
    return ret;
}

/** Reserve device quota atomically under the shared lock. */
static int reserve_device_memory_impl(CUdevice dev, size_t size) {
    lock_shrreg();
    if (oom_check_impl(dev, size, 1)) {
        unlock_shrreg();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    add_gpu_device_memory_usage(getpid(), dev, size, 2);
    unlock_shrreg();
    return 0;
}

/** Refresh graph backing before atomically reserving ordinary device quota. */
int reserve_device_memory(CUdevice dev, size_t size) {
    graph_reclaim(0);
    int ret = reserve_device_memory_impl(dev, size);
    if (ret != CUDA_SUCCESS) {
        unsigned int trim_mask = graph_trim_mask(dev, size);
        if (trim_mask != 0) graph_reclaim(trim_mask);
        ret = reserve_device_memory_impl(dev, size);
    }
    return ret;
}

void release_device_memory(CUdevice dev, size_t size) {
    lock_shrreg();
    rm_gpu_device_memory_usage(getpid(), dev, size, 2);
    unlock_shrreg();
}

static CUresult reserve_module_memory_impl(CUdevice dev, size_t size) {
    lock_shrreg();
    if (oom_check_impl(dev, size, 1)) {
        unlock_shrreg();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    add_gpu_device_memory_usage(getpid(), dev, size, 1);
    unlock_shrreg();
    return CUDA_SUCCESS;
}

CUresult reserve_module_memory(CUdevice dev, size_t size) {
    graph_reclaim(0);
    CUresult ret = reserve_module_memory_impl(dev, size);
    if (ret != CUDA_SUCCESS) {
        unsigned int trim_mask = graph_trim_mask(dev, size);
        if (trim_mask != 0) graph_reclaim(trim_mask);
        ret = reserve_module_memory_impl(dev, size);
    }
    return ret;
}

void release_module_memory(CUdevice dev, size_t size) {
    lock_shrreg();
    rm_gpu_device_memory_usage(getpid(), dev, size, 1);
    unlock_shrreg();
}

static pthread_mutex_t mem_op_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t mem_op_cond = PTHREAD_COND_INITIALIZER;
static pthread_once_t mem_op_atfork_once = PTHREAD_ONCE_INIT;
static int mem_op_inflight = 0;
static int mem_op_sampling = 0;

static void reset_memory_ops_after_fork(void) {
    mem_op_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    mem_op_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
    mem_op_inflight = 0;
    mem_op_sampling = 0;
}

static void register_memory_ops_atfork(void) {
    pthread_atfork(NULL, NULL, reset_memory_ops_after_fork);
}

void memory_op_enter(void) {
    pthread_once(&mem_op_atfork_once, register_memory_ops_atfork);
    pthread_mutex_lock(&mem_op_mutex);
    while (mem_op_sampling > 0) {
        pthread_cond_wait(&mem_op_cond, &mem_op_mutex);
    }
    mem_op_inflight++;
    pthread_mutex_unlock(&mem_op_mutex);
}

void memory_op_leave(void) {
    pthread_mutex_lock(&mem_op_mutex);
    mem_op_inflight--;
    if (mem_op_inflight == 0 && mem_op_sampling > 0) {
        pthread_cond_broadcast(&mem_op_cond);
    }
    pthread_mutex_unlock(&mem_op_mutex);
}

void memory_sampling_enter(void) {
    pthread_once(&mem_op_atfork_once, register_memory_ops_atfork);
    pthread_mutex_lock(&mem_op_mutex);
    mem_op_sampling++;
    while (mem_op_inflight > 0) {
        pthread_cond_wait(&mem_op_cond, &mem_op_mutex);
    }
    pthread_mutex_unlock(&mem_op_mutex);
}

void memory_sampling_leave(void) {
    pthread_mutex_lock(&mem_op_mutex);
    mem_op_sampling--;
    if (mem_op_sampling == 0) {
        pthread_cond_broadcast(&mem_op_cond);
    }
    pthread_mutex_unlock(&mem_op_mutex);
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
struct vmm_map_entry {
    CUdeviceptr ptr;
    vmm_handle_entry *h;
    struct vmm_map_entry *next;
};
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

/* Called before the driver cuMemMap. For a tracked handle, takes a temporary
 * reference so a concurrent release cannot drop the charge before the mapping
 * is recorded. *out stays NULL for untracked handles. Returns -1 on OOM. */
int vmm_map_begin(CUmemGenericAllocationHandle handle, vmm_map_entry **out) {
    vmm_map_entry *m = calloc(1, sizeof(*m));
    int ret = 0;
    *out = NULL;
    pthread_mutex_lock(&vmm_mutex);
    vmm_handle_entry *h = vmm_find(handle);
    if (h != NULL && m != NULL) {
        h->refs++;
        m->h = h;
        *out = m;
        m = NULL;
    } else if (h != NULL) {
        ret = -1;
    }
    pthread_mutex_unlock(&vmm_mutex);
    free(m);
    return ret;
}

/* Called after the driver cuMemMap: records the mapping if it succeeded and
 * drops the temporary reference. */
void vmm_map_end(vmm_map_entry *m, CUdeviceptr ptr, int mapped) {
    if (m == NULL) return;
    pthread_mutex_lock(&vmm_mutex);
    vmm_handle_entry *h = m->h;
    if (mapped) {
        m->ptr = ptr;
        m->next = vmm_maps;
        vmm_maps = m;
        h->nmaps++;
        m = NULL;
    }
    h->refs--;
    vmm_maybe_drop(h);
    pthread_mutex_unlock(&vmm_mutex);
    free(m);
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

/** Keep asynchronous free tracking within the capture transition lock. */
int free_raw_async(CUdeviceptr dptr, CUstream hStream) {
    pthread_mutex_lock(&mutex);
    unsigned int tmp = remove_chunk_async(device_allocasync, dptr, hStream);
    if (tmp == CUDA_SUCCESS) graph_ptr_freed_async(dptr, hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

typedef struct graph_group {
    CUgraph graph;
    CUgraph owner;
    uint64_t root;
    int refs;
    size_t total[CUDA_DEVICE_MAX_COUNT];
    size_t temporary[CUDA_DEVICE_MAX_COUNT];
    size_t pending[CUDA_DEVICE_MAX_COUNT];
    struct graph_group *next;
} graph_group;

struct graph_charge_entry {
    CUgraph graph;       /* NULL once the graph is destroyed */
    CUdevice dev;
    size_t size;
    CUdeviceptr ptr;
    cuuint64_t capid;    /* stream capture it was recorded in, 0 for explicit nodes */
    int capturing;       /* that capture has not ended yet */
    int has_free;
    int nexec;           /* live execs referencing the allocation */
    int launched;        /* an exec of the graph was launched or uploaded */
    int launching;       /* admitted submissions not yet returned by the driver */
    int resident;
    int held;
    uint64_t freed;
    uint64_t generation;
    pthread_mutex_t lifetime_mutex;
    graph_group *group;
    struct graph_charge_entry *next;
};
typedef struct graph_exec_link {
    graph_charge_entry *entry;
    uint64_t freed;
    uint64_t generation;
    uint64_t saved_freed, saved_generation;
    int has_free, is_free, free_done, saved_free_done;
    struct graph_exec_link *update_next;
    struct graph_exec_link *next;
} graph_exec_link;

typedef struct graph_stream_key {
    CUcontext ctx;
    cuuint64_t ctx_id;
    CUstream stream;
    cuuint64_t id;
    uint64_t thread;
    uint64_t epoch;
    int native_id;
    int known;
} graph_stream_key;

struct graph_exec_state {
    CUgraphExec exec;
    CUevent done;
    CUcontext event_ctx;
    cuuint64_t event_ctx_id;
    CUevent prepared_event;
    CUevent free_event, prepared_free_event;
    CUcontext free_event_ctx;
    cuuint64_t free_event_ctx_id;
    int free_recorded, upload;
    graph_stream_key key, saved_key;
    graph_stream_key free_key;
    graph_group *group;
    graph_exec_link *entries;
    graph_exec_link *updates;
    pthread_mutex_t submit_mutex;
    int users;
    int entries_released;
    int launched;
    int has_frees;
    int fast;
    int was_active;
    int context_dead;
    int active;
    int recorded;
    int pending;
    int retired;
    uint64_t submitted;
    uint64_t free_submitted;
    size_t temporary[CUDA_DEVICE_MAX_COUNT];
    struct graph_exec_state *active_next;
    struct graph_exec_state *next;
};
typedef struct capture_link {
    CUstream stream;
    CUgraph graph;
    CUcontext ctx;
    cuuint64_t ctx_id;
    cuuint64_t id;
    uint64_t thread;
    uint64_t token;
    CUstreamCaptureMode mode;
    struct capture_link *next;
} capture_link;
static capture_link *capture_links;
static uint64_t capture_serial;
static graph_charge_entry *graph_charges;
static graph_group *graph_groups;
static graph_exec_state *graph_states;
static graph_exec_state *graph_active;
static graph_exec_state *graph_last_exec;
static pthread_mutex_t graph_mutex = PTHREAD_MUTEX_INITIALIZER;

static size_t graph_charged[CUDA_DEVICE_MAX_COUNT];
static size_t graph_reserved[CUDA_DEVICE_MAX_COUNT];
static int graph_submitted[CUDA_DEVICE_MAX_COUNT];
static size_t graph_total[CUDA_DEVICE_MAX_COUNT];
static size_t graph_persistent[CUDA_DEVICE_MAX_COUNT];
static size_t graph_pending_peak[CUDA_DEVICE_MAX_COUNT];
static size_t graph_active_bytes[CUDA_DEVICE_MAX_COUNT];
static unsigned int graph_pending_dirty;
static int graph_active_dirty;
static uint64_t graph_thread_serial, graph_stream_epoch;
static uint64_t graph_submit_serial;
static uint64_t graph_root_serial;
static _Thread_local uint64_t graph_thread_id;
static CUcontext graph_primary_ctx[CUDA_DEVICE_MAX_COUNT];
static cuuint64_t graph_primary_ctx_id[CUDA_DEVICE_MAX_COUNT];

static void graph_exec_maybe_free(graph_exec_state *s);
static int graph_query_begin(CUstreamCaptureMode *mode);
static void graph_query_end(CUstreamCaptureMode mode);

/** Release a graph group after its final reference is dropped. */
static void graph_group_put(graph_group *group) {
    if (group == NULL || --group->refs != 0) return;
    graph_group **pp = &graph_groups;
    while (*pp != group) pp = &(*pp)->next;
    *pp = group->next;
    free(group);
}

/** Resolve a live root without reusing a retired graph handle's identity. */
static uint64_t graph_root_id(CUgraph owner) {
    if (owner == NULL) return 0;
    for (graph_group *g = graph_groups; g != NULL; g = g->next) {
        if (g->owner == owner) return g->root;
    }
    return 0;
}

/** Sum pending allocations sharing a stable root identity. */
static size_t graph_pending_total(uint64_t root, int dev) {
    size_t total = 0;
    for (graph_group *g = graph_groups; g != NULL; g = g->next) {
        if (g->root == root) total += g->pending[dev];
    }
    return total;
}

/** Add a pending allocation and update its owning root's cached demand. */
static void graph_pending_add(graph_group *group, int dev, size_t bytes) {
    group->pending[dev] += bytes;
    size_t total = graph_pending_total(group->root, dev);
    if (total > graph_pending_peak[dev]) graph_pending_peak[dev] = total;
}

/** Remove a pending allocation and invalidate its cached peak when needed. */
static void graph_pending_remove(graph_group *group, int dev, size_t bytes) {
    if (graph_pending_total(group->root, dev) == graph_pending_peak[dev]) graph_pending_dirty |= 1u << dev;
    group->pending[dev] -= bytes;
}

/** Move an allocation's charge between pending and persistent demand. */
static void graph_entry_update(graph_charge_entry *e) {
    int resident = e->launched || e->launching;
    int held = resident || e->freed == 0;
    if (resident == e->resident && held == e->held) return;
    if (e->held) {
        if (e->resident) graph_persistent[e->dev] -= e->size;
        else graph_pending_remove(e->group, e->dev, e->size);
    }
    if (held) {
        if (resident) graph_persistent[e->dev] += e->size;
        else graph_pending_add(e->group, e->dev, e->size);
    }
    e->resident = resident;
    e->held = held;
}

/** Retire confirmed external frees without releasing a newer allocation generation. */
static int graph_complete_frees(graph_exec_state *s, int completed) {
    int pending = 0;
    for (graph_exec_link *l = s->updates; l != NULL; l = l->update_next) {
        if (!l->is_free || l->free_done) continue;
        if (l->entry->launching) return 0;
        pending = 1;
    }
    if (pending && !completed) {
        if (!s->free_recorded) return 0;
        for (capture_link *l = capture_links; l != NULL; l = l->next) {
            if (l->mode == CU_STREAM_CAPTURE_MODE_GLOBAL || l->thread == graph_thread_id) return 0;
        }
        CUstreamCaptureMode mode;
        if (!graph_query_begin(&mode)) return 0;
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventQuery, s->free_event);
        graph_query_end(mode);
        if (res != CUDA_SUCCESS) return 0;
    }
    for (graph_exec_link *l = s->updates; l != NULL; l = l->update_next) {
        if (!l->is_free || l->free_done) continue;
        graph_charge_entry *e = l->entry;
        if (e->generation == l->generation && e->freed == l->freed) {
            e->launched = 0;
            e->freed++;
            graph_entry_update(e);
            for (graph_exec_state *other = graph_states; other != NULL; other = other->next) {
                for (graph_exec_link *a = other->entries; a != NULL; a = a->next) {
                    if (a->entry == e && !a->is_free) other->launched = 0;
                }
            }
        }
        l->free_done = 1;
    }
    return 1;
}

/** Compare stream identities while accounting for context, thread, and handle reuse. */
static int graph_stream_equal(const graph_stream_key *a, const graph_stream_key *b) {
    if (!a->known || !b->known || a->ctx != b->ctx || a->ctx_id != b->ctx_id ||
        a->native_id != b->native_id) return 0;
    if (a->native_id) return a->id == b->id;
    return a->stream == b->stream && a->thread == b->thread && a->epoch == b->epoch;
}

/** Resolve stream identity across contexts, threads, and handle reuse. */
static CUresult graph_stream_info(CUstream stream, graph_stream_key *key) {
    memset(key, 0, sizeof(*key));
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamGetCtx, stream, &key->ctx);
    if (res != CUDA_SUCCESS) return res;
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuCtxGetId) != NULL) {
        res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetId, key->ctx, &key->ctx_id);
        if (res != CUDA_SUCCESS) return res;
    }
    key->stream = stream == NULL ? CU_STREAM_LEGACY : stream;
    if (key->stream == CU_STREAM_PER_THREAD) {
        if (graph_thread_id == 0) graph_thread_id = ++graph_thread_serial;
        key->thread = graph_thread_id;
    }
    key->known = 1;
    key->native_id = CUDA_FIND_ENTRY(cuda_library_entry, cuStreamGetId) != NULL &&
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamGetId, stream, &key->id) == CUDA_SUCCESS;
    return CUDA_SUCCESS;
}

/** Track an executable graph as active until its submitted GPU work completes. */
static void graph_active_add(graph_exec_state *s) {
    if (s->active) return;
    s->active = 1;
    s->active_next = graph_active;
    graph_active = s;
    graph_active_dirty = 1;
}

/** Remove an executable graph from the active list and invalidate cached demand. */
static void graph_active_remove(graph_exec_state *s) {
    if (!s->active) return;
    graph_exec_state **pp = &graph_active;
    while (*pp != s) pp = &(*pp)->active_next;
    *pp = s->active_next;
    s->active = 0;
    s->active_next = NULL;
    graph_active_dirty = 1;
}

/** Relax only internal queries and retain the caller's interaction mode. */
static int graph_query_begin(CUstreamCaptureMode *mode) {
    *mode = CU_STREAM_CAPTURE_MODE_RELAXED;
    return CUDA_FIND_ENTRY(cuda_library_entry, cuThreadExchangeStreamCaptureMode) != NULL &&
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuThreadExchangeStreamCaptureMode, mode) == CUDA_SUCCESS;
}

/** Restore the capture interaction mode after internal driver queries. */
static void graph_query_end(CUstreamCaptureMode mode) {
    CUDA_OVERRIDE_CALL(cuda_library_entry, cuThreadExchangeStreamCaptureMode, &mode);
}

/** Retire completed work while suppressing queries during capture. */
static void graph_poll_active(void) {
    /* GLOBAL capture protects every thread until native capture termination. */
    for (capture_link *l = capture_links; l != NULL; l = l->next) {
        if (l->mode == CU_STREAM_CAPTURE_MODE_GLOBAL || l->thread == graph_thread_id) return;
    }
    CUstreamCaptureMode mode;
    if (graph_active == NULL || !graph_query_begin(&mode)) return;
    for (graph_exec_state *s = graph_active, *next; s != NULL; s = next) {
        next = s->active_next;
        if (s->active && s->recorded && !s->pending &&
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventQuery, s->done) == CUDA_SUCCESS &&
            graph_complete_frees(s, 0))
            graph_active_remove(s);
        graph_exec_maybe_free(s);
    }
    graph_query_end(mode);
}

/** Refresh retained graph-pool backing and conservatively preserve charges on query failure. */
static void graph_refresh_pool(unsigned int trim_mask) {
    for (capture_link *l = capture_links; l != NULL; l = l->next) {
        if (l->mode == CU_STREAM_CAPTURE_MODE_GLOBAL || l->thread == graph_thread_id) return;
    }
    CUstreamCaptureMode mode;
    int relaxed = 0;
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        int tracked = graph_total[dev] != 0;
        if (!tracked && graph_charged[dev] == 0) continue;
        if (!relaxed && !graph_query_begin(&mode)) {
            for (int d = 0; d < CUDA_DEVICE_MAX_COUNT; d++) {
                if (graph_submitted[d] && graph_charged[d] > graph_reserved[d])
                    graph_reserved[d] = graph_charged[d];
            }
            return;
        }
        relaxed = 1;
        /* Release unused backing only when admission needs more space. */
        if ((trim_mask & (1u << dev)) && CUDA_FIND_ENTRY(cuda_library_entry, cuDeviceGraphMemTrim) != NULL)
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGraphMemTrim, dev);
        cuuint64_t reserved;
        if (CUDA_FIND_ENTRY(cuda_library_entry, cuDeviceGetGraphMemAttribute) != NULL &&
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetGraphMemAttribute, dev,
                CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT, &reserved) == CUDA_SUCCESS) {
            graph_reserved[dev] = reserved;
        } else if (graph_submitted[dev] && graph_charged[dev] > graph_reserved[dev]) {
            /* A failed query cannot establish that cached backing was freed. */
            graph_reserved[dev] = graph_charged[dev];
        }
    }
    if (relaxed) graph_query_end(mode);
}

static void graph_refresh(void) {
    graph_poll_active();
    graph_refresh_pool(0);
}

/** Refresh cached peaks only when graph or execution state changes. */
static void graph_update_cache(void) {
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        if (!(graph_pending_dirty & (1u << dev))) continue;
        graph_pending_peak[dev] = 0;
        for (graph_group *g = graph_groups; g != NULL; g = g->next) {
            size_t total = graph_pending_total(g->root, dev);
            if (total > graph_pending_peak[dev]) graph_pending_peak[dev] = total;
        }
    }
    graph_pending_dirty = 0;
    if (!graph_active_dirty) return;
    memset(graph_active_bytes, 0, sizeof(graph_active_bytes));
    for (graph_exec_state *s = graph_active; s != NULL; s = s->active_next) {
        graph_exec_state *o;
        for (o = graph_active; o != s; o = o->active_next) {
            if (graph_stream_equal(&s->key, &o->key)) break;
        }
        if (o != s) continue;
        size_t peak[CUDA_DEVICE_MAX_COUNT];
        memcpy(peak, s->temporary, sizeof(peak));
        for (o = s->active_next; o != NULL; o = o->active_next) {
            if (!graph_stream_equal(&s->key, &o->key)) continue;
            for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
                if (o->temporary[dev] > peak[dev]) peak[dev] = o->temporary[dev];
            }
        }
        for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) graph_active_bytes[dev] += peak[dev];
    }
    graph_active_dirty = 0;
}

/** Combine cached pending and active graph demand. */
static size_t graph_logical_wanted(int dev) {
    size_t want = graph_persistent[dev] + graph_active_bytes[dev];
    if (graph_pending_peak[dev] > want) want = graph_pending_peak[dev];
    return want;
}

/** Include retained driver backing in the quota charge. */
static size_t graph_wanted(int dev) {
    size_t want = graph_logical_wanted(dev);
    return graph_reserved[dev] > want ? graph_reserved[dev] : want;
}

/** Trim only devices whose maximum refund can cover the quota shortfall. */
static unsigned int graph_trim_mask(int dev, size_t addon) {
    CUdevice current = dev;
    if (current == -1 && cuCtxGetDevice(&current) != CUDA_SUCCESS) return 0;
    if (current < 0 || current >= CUDA_DEVICE_MAX_COUNT) return 0;
    pthread_mutex_lock(&graph_mutex);
    graph_update_cache();
    size_t logical = graph_logical_wanted(current);
    size_t refundable = graph_charged[current] > logical ? graph_charged[current] - logical : 0;
    lock_shrreg();
    uint64_t limit = get_current_device_memory_limit(current);
    size_t usage = get_gpu_memory_usage(cuda_to_nvml_map(current));
    unlock_shrreg();
    unsigned int mask = 0;
    if (refundable != 0 && limit != 0 && (usage > limit || addon > limit - usage)) {
        size_t remaining = usage > refundable ? usage - refundable : 0;
        if (remaining <= limit && addon <= limit - remaining) mask = 1u << current;
    }
    pthread_mutex_unlock(&graph_mutex);
    return mask;
}

/** Apply cached graph demand changes under the shared quota lock. */
static int graph_rebalance(int check) {
    graph_update_cache();
    int ret = 0;
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        size_t want = graph_wanted(dev);
        size_t have = graph_charged[dev];
        if (want > have) {
            if (check) {
                if (reserve_device_memory_impl(dev, want - have) != 0) {
                    ret = CUDA_ERROR_OUT_OF_MEMORY;
                    continue;
                }
            } else {
                lock_shrreg();
                add_gpu_device_memory_usage(getpid(), dev, want - have, 2);
                unlock_shrreg();
            }
            graph_charged[dev] = want;
        } else if (want < have) {
            release_device_memory(dev, have - want);
            graph_charged[dev] = want;
        }
    }
    return ret;
}

static void graph_reclaim(unsigned int trim_mask) {
    pthread_mutex_lock(&graph_mutex);
    graph_poll_active();
    graph_refresh_pool(trim_mask);
    graph_rebalance(0);
    pthread_mutex_unlock(&graph_mutex);
}

void graph_pool_trimmed(void) {
    graph_reclaim(0);
}

/** Remove an allocation and its executable references from cached demand. */
static void graph_drop_entry(graph_charge_entry *e) {
    for (graph_exec_state *s = graph_states; s != NULL; s = s->next) {
        graph_exec_link **update = &s->updates;
        while (*update != NULL) {
            if ((*update)->entry == e) *update = (*update)->update_next;
            else update = &(*update)->update_next;
        }
        graph_exec_link **lp = &s->entries;
        while (*lp != NULL) {
            if ((*lp)->entry == e) {
                graph_exec_link *l = *lp;
                *lp = l->next;
                free(l);
            } else {
                lp = &(*lp)->next;
            }
        }
    }
    graph_charge_entry **pp = &graph_charges;
    while (*pp != e) pp = &(*pp)->next;
    *pp = e->next;
    graph_group *group = e->group;
    group->total[e->dev] -= e->size;
    graph_total[e->dev] -= e->size;
    if (e->has_free) group->temporary[e->dev] -= e->size;
    if (e->held) {
        if (e->resident) graph_persistent[e->dev] -= e->size;
        else graph_pending_remove(group, e->dev, e->size);
    }
    pthread_mutex_destroy(&e->lifetime_mutex);
    free(e);
    graph_group_put(group);
}

/** Reserve quota and bookkeeping before adding a graph allocation. */
int graph_alloc_begin(CUgraph graph, CUdevice dev, size_t size, cuuint64_t capid, int capturing,
                      graph_charge_entry **out) {
    *out = NULL;
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) return CUDA_SUCCESS;
    int count;
    CUresult valid = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetCount, &count);
    if (valid != CUDA_SUCCESS) return valid;
    if (dev >= count) return CUDA_SUCCESS;
    graph_charge_entry *g = calloc(1, sizeof(*g));
    if (g == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
    if (pthread_mutex_init(&g->lifetime_mutex, NULL) != 0) {
        free(g);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    g->graph = graph;
    g->dev = dev;
    g->size = size;
    g->held = 1;
    g->capid = capid;
    g->capturing = capturing;
    pthread_mutex_lock(&graph_mutex);
    graph_group *group;
    for (group = graph_groups; group != NULL; group = group->next) {
        if (group->graph == graph) break;
    }
    if (group == NULL) {
        group = calloc(1, sizeof(*group));
        if (group == NULL) {
            pthread_mutex_destroy(&g->lifetime_mutex);
            free(g);
            pthread_mutex_unlock(&graph_mutex);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        group->graph = graph;
        group->owner = graph;
        group->root = graph_root_id(graph);
        if (group->root == 0) group->root = ++graph_root_serial;
        group->next = graph_groups;
        graph_groups = group;
    }
    g->group = group;
    group->refs++;
    group->total[dev] += size;
    graph_total[dev] += size;
    graph_pending_add(group, dev, size);
    g->next = graph_charges;
    graph_charges = g;
    if (!capturing) graph_refresh();
    int ret = graph_rebalance(1);
    if (ret != 0) {
        graph_drop_entry(g);
        if (!capturing) graph_refresh();
        graph_rebalance(0);
    } else {
        *out = g;
    }
    pthread_mutex_unlock(&graph_mutex);
    return ret;
}

/** Commit a graph allocation or roll back its reservation after native failure. */
void graph_alloc_end(graph_charge_entry *entry, CUdeviceptr ptr, CUresult result) {
    if (entry == NULL) return;
    pthread_mutex_lock(&graph_mutex);
    if (result == CUDA_SUCCESS) {
        entry->ptr = ptr;
    } else {
        int capturing = entry->capturing;
        graph_drop_entry(entry);
        if (!capturing) graph_refresh();
        graph_rebalance(0);
    }
    pthread_mutex_unlock(&graph_mutex);
}

/** Record captured allocations against cached state until pre-admission is available. */
int graph_record(CUgraph graph, CUdevice dev, size_t size, CUdeviceptr ptr, cuuint64_t capid) {
    graph_charge_entry *entry;
    int ret = graph_alloc_begin(graph, dev, size, capid, 1, &entry);
    if (ret == CUDA_SUCCESS) graph_alloc_end(entry, ptr, CUDA_SUCCESS);
    return ret;
}

/** Record graph-owned frees without driver queries during capture. */
void graph_mark_free(CUgraph graph, CUdeviceptr ptr) {
    pthread_mutex_lock(&graph_mutex);
    for (graph_charge_entry *g = graph_charges; g != NULL; g = g->next) {
        if (g->ptr == ptr && g->graph == graph && !g->has_free) {
            g->has_free = 1;
            g->group->temporary[g->dev] += g->size;
            graph_entry_update(g);
        }
    }
    /* Recording a free node may run during capture, so defer driver queries. */
    graph_rebalance(0);
    pthread_mutex_unlock(&graph_mutex);
}

/** Update graph allocation lifetimes after an external pointer free. */
void graph_ptr_freed(CUdeviceptr ptr) {
    pthread_mutex_lock(&graph_mutex);
    for (graph_charge_entry *g = graph_charges, *n; g != NULL; g = n) {
        n = g->next;
        if (g->ptr != ptr) continue;
        if (g->graph == NULL && g->nexec == 0 && g->launching == 0) {
            graph_drop_entry(g);
        } else {
            g->launched = 0;
            g->freed++;
            graph_entry_update(g);
            for (graph_exec_state *s = graph_states; s != NULL; s = s->next) {
                for (graph_exec_link *l = s->entries; l != NULL; l = l->next) {
                    if (l->entry == g) s->launched = 0;
                }
            }
        }
    }
    graph_refresh();
    graph_rebalance(0);
    pthread_mutex_unlock(&graph_mutex);
}

/** Return native capture state while preserving query failures. */
CUresult graph_capture_info(CUstream hStream, CUstreamCaptureStatus *cap, cuuint64_t *id, CUgraph *graph) {
    *cap = CU_STREAM_CAPTURE_STATUS_NONE;
    *id = 0;
    *graph = NULL;
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuStreamGetCaptureInfo_v3) != NULL) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamGetCaptureInfo_v3, hStream, cap, id, graph,
                                   NULL, NULL, NULL);
    }
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamGetCaptureInfo_v2, hStream, cap, id, graph, NULL, NULL);
}

/** Register a capture scope before entering the native capture call. */
int graph_capture_prepare(CUstream stream, CUstreamCaptureMode mode, uint64_t *token) {
    capture_link *l = calloc(1, sizeof(*l));
    if (l == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
    l->stream = stream;
    l->mode = mode;
    if (stream == CU_STREAM_PER_THREAD) {
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetCurrent, &l->ctx);
        if (res != CUDA_SUCCESS) {
            free(l);
            return res;
        }
    } else if (CUDA_FIND_ENTRY(cuda_library_entry, cuStreamGetCtx) != NULL &&
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamGetCtx, stream, &l->ctx) != CUDA_SUCCESS) {
        l->ctx = NULL;
    }
    if (l->ctx != NULL) l->ctx_id = graph_context_id(l->ctx);
    pthread_mutex_lock(&graph_mutex);
    if (graph_thread_id == 0) graph_thread_id = ++graph_thread_serial;
    l->thread = graph_thread_id;
    l->token = ++capture_serial;
    *token = l->token;
    l->next = capture_links;
    capture_links = l;
    pthread_mutex_unlock(&graph_mutex);
    return CUDA_SUCCESS;
}

/** Remove the capture scope identified by its token. */
void graph_capture_finish(uint64_t token) {
    pthread_mutex_lock(&graph_mutex);
    capture_link **pp = &capture_links;
    while (*pp != NULL) {
        capture_link *l = *pp;
        if (l->token == token) {
            *pp = l->next;
            free(l);
            break;
        }
        pp = &l->next;
    }
    pthread_mutex_unlock(&graph_mutex);
}

/** Record a successful capture or discard its scope after native failure. */
void graph_capture_started(uint64_t token, CUresult result) {
    if (result != CUDA_SUCCESS) {
        graph_capture_finish(token);
        return;
    }
    pthread_mutex_lock(&graph_mutex);
    for (capture_link *l = capture_links; l != NULL; l = l->next) {
        if (l->token != token) continue;
        CUstreamCaptureStatus cap;
        cuuint64_t id = 0;
        CUgraph graph = NULL;
        if (graph_capture_info(l->stream, &cap, &id, &graph) == CUDA_SUCCESS &&
            cap != CU_STREAM_CAPTURE_STATUS_NONE) {
            l->id = id;
            l->graph = graph;
        }
        break;
    }
    pthread_mutex_unlock(&graph_mutex);
}

/** Resolve a capture identity before native termination can mutate its scope. */
CUresult graph_capture_resolve(CUstream stream, cuuint64_t *id, uint64_t *token) {
    *token = 0;
    *id = 0;
    CUcontext ctx = NULL;
    if (stream == CU_STREAM_PER_THREAD) {
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetCurrent, &ctx);
        if (res != CUDA_SUCCESS) return res;
    }
    pthread_mutex_lock(&graph_mutex);
    for (capture_link *l = capture_links; l != NULL; l = l->next) {
        if (l->stream != stream || (stream == CU_STREAM_PER_THREAD &&
            (l->thread != graph_thread_id || l->ctx != ctx))) continue;
        *token = l->token;
        *id = l->id;
        if (l->id != 0) break;
    }
    pthread_mutex_unlock(&graph_mutex);
    return CUDA_SUCCESS;
}

/** Return the tracked capture token for callers that do not mutate native state. */
uint64_t graph_capture_token(CUstream stream, cuuint64_t *id) {
    uint64_t token;
    graph_capture_resolve(stream, id, &token);
    return token;
}


cuuint64_t graph_capture_id_of(CUgraph graph) {
    cuuint64_t id = 0;
    pthread_mutex_lock(&graph_mutex);
    for (capture_link *l = capture_links; l != NULL; l = l->next) {
        if (l->graph == graph) {
            id = l->id;
            break;
        }
    }
    pthread_mutex_unlock(&graph_mutex);
    return id;
}

/** Record a captured free node or release the pointer's ordinary graph charge. */
void graph_ptr_freed_async(CUdeviceptr ptr, CUstream hStream) {
    CUstreamCaptureStatus cap;
    cuuint64_t id;
    CUgraph graph;
    graph_capture_info(hStream, &cap, &id, &graph);
    if (cap != CU_STREAM_CAPTURE_STATUS_NONE) {
        graph_mark_free(graph, ptr);
    } else {
        graph_ptr_freed(ptr);
    }
}

/** Retain an allocation or external free in the executable's immutable snapshot. */
static int graph_exec_link_entry(graph_exec_state *s, graph_charge_entry *e, int is_free) {
    graph_exec_link *l = calloc(1, sizeof(*l));
    if (l == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
    l->entry = e;
    l->has_free = e->has_free;
    l->is_free = is_free;
    l->free_done = 1;
    l->next = s->entries;
    s->entries = l;
    e->nexec++;
    return CUDA_SUCCESS;
}

/** Read free nodes recursively before instantiation can make their execution possible. */
static CUresult graph_exec_collect_frees(graph_exec_state *s, CUgraph graph) {
    size_t count = 0;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphGetNodes, graph, NULL, &count);
    if (res != CUDA_SUCCESS || count == 0) return res;
    if (count > SIZE_MAX / sizeof(CUgraphNode)) return CUDA_ERROR_OUT_OF_MEMORY;
    size_t capacity = count;
    CUgraphNode *nodes = calloc(count, sizeof(*nodes));
    if (nodes == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphGetNodes, graph, nodes, &count);
    if (res == CUDA_SUCCESS && count > capacity) res = CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < count && res == CUDA_SUCCESS; i++) {
        CUgraphNodeType type;
        res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphNodeGetType, nodes[i], &type);
        if (res != CUDA_SUCCESS) break;
        if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
            CUgraph child;
            res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphChildGraphNodeGetGraph, nodes[i], &child);
            if (res == CUDA_SUCCESS) res = graph_exec_collect_frees(s, child);
        } else if (type == CU_GRAPH_NODE_TYPE_MEM_FREE) {
            CUdeviceptr ptr;
            res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphMemFreeNodeGetParams, nodes[i], &ptr);
            if (res != CUDA_SUCCESS) break;
            graph_exec_link *local;
            for (local = s->entries; local != NULL; local = local->next) {
                if (local->entry->ptr == ptr) break;
            }
            if (local != NULL) {
                if (!local->is_free) local->has_free = 1;
                continue;
            }
            for (graph_charge_entry *e = graph_charges; e != NULL; e = e->next) {
                if (e->ptr != ptr) continue;
                res = graph_exec_link_entry(s, e, 1);
                break;
            }
        }
    }
    free(nodes);
    return res;
}

/** Reserve combined pending demand before moving a child into its parent. */
int graph_child_begin(CUgraph parent, CUgraph child) {
    pthread_mutex_lock(&graph_mutex);
    graph_refresh();
    graph_update_cache();
    uint64_t parent_root = graph_root_id(parent), child_root = graph_root_id(child);
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        size_t total = graph_pending_total(parent_root, dev);
        if (parent_root != child_root) total += graph_pending_total(child_root, dev);
        if (total > graph_pending_peak[dev]) graph_pending_peak[dev] = total;
    }
    int ret = graph_rebalance(1);
    if (ret != CUDA_SUCCESS) {
        graph_pending_dirty = (1u << CUDA_DEVICE_MAX_COUNT) - 1;
        graph_rebalance(0);
        pthread_mutex_unlock(&graph_mutex);
    }
    return ret;
}

/** Commit child ownership or restore the reservation after native failure. */
void graph_child_end(CUgraph parent, CUgraph child, CUresult result) {
    uint64_t root = graph_root_id(parent);
    for (graph_group *g = graph_groups; g != NULL; g = g->next) {
        if (result == CUDA_SUCCESS && g->owner == child) {
            if (root == 0) root = ++graph_root_serial;
            g->owner = parent;
            g->root = root;
        }
    }
    graph_pending_dirty = (1u << CUDA_DEVICE_MAX_COUNT) - 1;
    graph_rebalance(0);
    pthread_mutex_unlock(&graph_mutex);
}

/** Cache graph demand and lifetime classifications before native executable instantiation. */
int graph_exec_prepare(CUgraph graph, graph_exec_state **out) {
    *out = NULL;
    if (graph == NULL) return 0;
    pthread_mutex_lock(&graph_mutex);
    graph_group *group;
    if (graph_charges == NULL) {
        pthread_mutex_unlock(&graph_mutex);
        return 0;
    }
    for (group = graph_groups; group != NULL; group = group->next) {
        if (group->owner == graph) break;
    }
    CUresult res = CUDA_ERROR_OUT_OF_MEMORY;
    graph_exec_state *s = calloc(1, sizeof(*s));
    if (s == NULL) goto fail;
    if (pthread_mutex_init(&s->submit_mutex, NULL) != 0) {
        free(s);
        s = NULL;
        goto fail;
    }
    s->group = group;
    if (group != NULL) group->refs++;
    for (graph_charge_entry *g = graph_charges; g != NULL; g = g->next) {
        if (g->graph == NULL || g->group->owner != graph) continue;
        if (graph_exec_link_entry(s, g, 0) != CUDA_SUCCESS) goto fail;
    }
    res = graph_exec_collect_frees(s, graph);
    if (res != CUDA_SUCCESS) goto fail;
    if (s->entries == NULL) {
        graph_group_put(group);
        pthread_mutex_destroy(&s->submit_mutex);
        free(s);
        pthread_mutex_unlock(&graph_mutex);
        return CUDA_SUCCESS;
    }
    for (graph_exec_link *l = s->entries; l != NULL; l = l->next) {
        if (l->is_free || !l->has_free) {
            graph_exec_link **position = &s->updates;
            while (*position != NULL && (uintptr_t)(*position)->entry < (uintptr_t)l->entry)
                position = &(*position)->update_next;
            l->update_next = *position;
            *position = l;
            if (l->is_free) s->has_frees = 1;
        } else {
            s->temporary[l->entry->dev] += l->entry->size;
        }
    }
    s->next = graph_states;
    graph_states = s;
    *out = s;
    pthread_mutex_unlock(&graph_mutex);
    return 0;
fail:
    if (s != NULL) {
        while (s->entries != NULL) {
            graph_exec_link *l = s->entries;
            s->entries = l->next;
            l->entry->nexec--;
            free(l);
        }
        graph_group_put(group);
        pthread_mutex_destroy(&s->submit_mutex);
        free(s);
    }
    pthread_mutex_unlock(&graph_mutex);
    return res;
}

/** Associate prepared accounting state with a successfully instantiated executable graph. */
void graph_exec_commit(graph_exec_state *s, CUgraphExec exec) {
    if (s == NULL) return;
    pthread_mutex_lock(&graph_mutex);
    s->exec = exec;
    graph_last_exec = s;
    pthread_mutex_unlock(&graph_mutex);
}

/** Retain retired metadata until host submissions and GPU work finish. */
static void graph_exec_maybe_free(graph_exec_state *s) {
    if (!s->retired || s->users || s->pending || s->active) return;
    if (!s->entries_released) {
        s->entries_released = 1;
        s->updates = NULL;
        while (s->entries != NULL) {
            graph_exec_link *l = s->entries;
            graph_charge_entry *e = l->entry;
            s->entries = l->next;
            free(l);
            if (--e->nexec == 0 && e->graph == NULL && !e->launching &&
                !e->launched) graph_drop_entry(e);
        }
    }
    if (s->done != NULL) CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->done);
    if (s->prepared_event != NULL)
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->prepared_event);
    if (s->free_event != NULL) CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->free_event);
    if (s->prepared_free_event != NULL)
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->prepared_free_event);
    graph_exec_state **pp = &graph_states;
    while (*pp != s) pp = &(*pp)->next;
    *pp = s->next;
    if (graph_last_exec == s) graph_last_exec = NULL;
    graph_group_put(s->group);
    pthread_mutex_destroy(&s->submit_mutex);
    free(s);
}

/** Retire an executable graph while preserving charges for outstanding work. */
static void graph_exec_retire(graph_exec_state *s) {
    s->retired = 1;
    if (graph_last_exec == s) graph_last_exec = NULL;
    graph_exec_maybe_free(s);
    graph_refresh();
    graph_rebalance(0);
}

/** Discard prepared executable-graph state while respecting outstanding references. */
void graph_exec_discard(graph_exec_state *s) {
    if (s == NULL) return;
    pthread_mutex_lock(&graph_mutex);
    graph_exec_retire(s);
    pthread_mutex_unlock(&graph_mutex);
}

/** Find a live executable graph, reusing the most recent lookup when possible. */
static graph_exec_state *graph_exec_find(CUgraphExec exec) {
    if (exec == NULL) return NULL;
    if (graph_last_exec != NULL && !graph_last_exec->retired && graph_last_exec->exec == exec)
        return graph_last_exec;
    for (graph_exec_state *s = graph_states; s != NULL; s = s->next) {
        if (!s->retired && s->exec == exec) {
            graph_last_exec = s;
            return s;
        }
    }
    return NULL;
}

/** Retire accounting state after native executable-graph destruction succeeds. */
void graph_exec_destroyed(CUgraphExec exec) {
    pthread_mutex_lock(&graph_mutex);
    graph_exec_state *s = graph_exec_find(exec);
    if (s != NULL) graph_exec_retire(s);
    pthread_mutex_unlock(&graph_mutex);
}

/** Create a completion event in the submission context and restore the caller's context. */
static CUresult graph_new_event(CUcontext ctx, CUevent *event) {
    CUcontext current, popped;
    *event = NULL;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetCurrent, &current);
    if (res != CUDA_SUCCESS) return res;
    if (current != ctx) {
        res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxPushCurrent_v2, ctx);
        if (res != CUDA_SUCCESS) return res;
    }
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventCreate, event, CU_EVENT_DISABLE_TIMING);
    if (res != CUDA_SUCCESS) *event = NULL;
    if (current != ctx) {
        CUresult pop = CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxPopCurrent_v2, &popped);
        if (res == CUDA_SUCCESS && pop != CUDA_SUCCESS) {
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, *event);
            *event = NULL;
            res = pop;
        }
    }
    return res;
}

/** Report whether an executable graph contains temporary allocations. */
static int graph_has_temporary(graph_exec_state *s) {
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        if (s->temporary[dev] != 0) return 1;
    }
    return 0;
}

/** Snapshot external frees and protect persistent allocations during native submission. */
static void graph_prepare_entries(graph_exec_state *s) {
    for (graph_exec_link *l = s->updates; l != NULL; l = l->update_next) {
        graph_charge_entry *e = l->entry;
        if (l->is_free) {
            if (s->upload) continue;
            l->saved_freed = l->freed;
            l->saved_generation = l->generation;
            l->saved_free_done = l->free_done;
            l->freed = e->freed;
            l->generation = e->generation;
            l->free_done = 0;
        } else {
            l->freed = e->freed;
            e->launching++;
            graph_entry_update(e);
        }
    }
}

/** Restore entry state when submission admission or the native operation fails. */
static void graph_cancel_entries(graph_exec_state *s) {
    for (graph_exec_link *l = s->updates; l != NULL; l = l->update_next) {
        if (l->is_free) {
            if (s->upload) continue;
            l->freed = l->saved_freed;
            l->generation = l->saved_generation;
            l->free_done = l->saved_free_done;
        } else {
            l->entry->launching--;
            graph_entry_update(l->entry);
        }
    }
}

/** Reserve quota and completion tracking for a pending graph submission. */
static int graph_exec_reserve_locked(graph_exec_state *s, CUstream stream, int upload) {
    int temporary = graph_has_temporary(s) || s->has_frees;
    graph_stream_key key = {0};
    if (temporary) {
        CUresult res = graph_stream_info(stream, &key);
        if (res != CUDA_SUCCESS) return res;
    }
    s->fast = s->launched && (!temporary ||
        (s->active && s->done != NULL && s->event_ctx == key.ctx && s->event_ctx_id == key.ctx_id &&
         graph_stream_equal(&s->key, &key)));
    if (!s->fast) graph_poll_active();
    s->was_active = s->active;
    s->saved_key = s->key;
    s->context_dead = 0;
    if (temporary && (s->done == NULL || s->event_ctx != key.ctx || s->event_ctx_id != key.ctx_id)) {
        CUresult res = graph_new_event(key.ctx, &s->prepared_event);
        if (res != CUDA_SUCCESS) return res;
    }
    if (!upload && s->has_frees && (s->free_event == NULL ||
        s->free_event_ctx != key.ctx || s->free_event_ctx_id != key.ctx_id)) {
        CUresult res = graph_new_event(key.ctx, &s->prepared_free_event);
        if (res != CUDA_SUCCESS) {
            if (s->prepared_event != NULL) {
                CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->prepared_event);
                s->prepared_event = NULL;
            }
            return res;
        }
    }
    s->upload = upload;
    s->pending = 1;
    graph_prepare_entries(s);
    if (s->fast) return 0;
    if (temporary) {
        if (s->was_active && !graph_stream_equal(&s->key, &key)) key.known = 0;
        s->key = key;
        graph_active_add(s);
        graph_active_dirty = 1;
    }
    graph_refresh_pool(0);
    int ret = graph_rebalance(1);
    if (ret != 0) {
        graph_cancel_entries(s);
        if (!s->was_active) graph_active_remove(s);
        s->key = s->saved_key;
        graph_active_dirty = 1;
        s->pending = 0;
        if (s->prepared_event != NULL) {
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->prepared_event);
            s->prepared_event = NULL;
        }
        if (s->prepared_free_event != NULL) {
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->prepared_free_event);
            s->prepared_free_event = NULL;
        }
        graph_rebalance(0);
    }
    return ret;
}

/** Drop a host reference and reclaim retired executable-graph state when possible. */
static void graph_exec_put_user(graph_exec_state *s) {
    pthread_mutex_lock(&graph_mutex);
    s->users--;
    if (s->retired) {
        graph_exec_maybe_free(s);
        graph_refresh();
        graph_rebalance(0);
    }
    pthread_mutex_unlock(&graph_mutex);
}

/** Release pointer locks after native submission state has been committed. */
static void graph_unlock_entries(graph_exec_state *s) {
    for (graph_exec_link *l = s->updates; l != NULL; l = l->update_next)
        pthread_mutex_unlock(&l->entry->lifetime_mutex);
}

/** Serialize submissions without holding the global graph lock while waiting. */
static int graph_exec_lock_and_reserve(graph_exec_state *s, CUstream stream, int upload) {
    pthread_mutex_lock(&s->submit_mutex);
    for (graph_exec_link *l = s->updates; l != NULL; l = l->update_next)
        pthread_mutex_lock(&l->entry->lifetime_mutex);
    pthread_mutex_lock(&graph_mutex);
    int ret = s->retired ? CUDA_ERROR_INVALID_HANDLE : graph_exec_reserve_locked(s, stream, upload);
    pthread_mutex_unlock(&graph_mutex);
    if (ret != 0) {
        graph_unlock_entries(s);
        pthread_mutex_unlock(&s->submit_mutex);
        graph_exec_put_user(s);
    }
    return ret;
}

/** Hold an executable-graph reference and reserve its submission quota. */
int graph_exec_reserve(graph_exec_state *s, CUstream stream, int upload) {
    if (s == NULL) return 0;
    pthread_mutex_lock(&graph_mutex);
    s->users++;
    pthread_mutex_unlock(&graph_mutex);
    return graph_exec_lock_and_reserve(s, stream, upload);
}

/** Find an executable graph and reserve its next submission before native execution. */
int graph_exec_begin(CUgraphExec exec, CUstream stream, int upload, graph_exec_state **out) {
    *out = NULL;
    pthread_mutex_lock(&graph_mutex);
    graph_exec_state *s = graph_exec_find(exec);
    if (s != NULL) s->users++;
    pthread_mutex_unlock(&graph_mutex);
    if (s == NULL) return 0;
    int ret = graph_exec_lock_and_reserve(s, stream, upload);
    if (ret == 0) *out = s;
    return ret;
}

/** Commit or roll back submission charges and update GPU completion tracking. */
void graph_exec_end(graph_exec_state *s, CUstream stream, CUresult result) {
    if (s == NULL) return;
    pthread_mutex_lock(&graph_mutex);
    /* Count submissions after native return so synchronization snapshots exclude pending calls. */
    if (result == CUDA_SUCCESS) s->submitted = ++graph_submit_serial;
    if (result == CUDA_SUCCESS) {
        s->launched = 1;
        for (graph_exec_link *l = s->updates; l != NULL; l = l->update_next) {
            if (l->is_free) continue;
            graph_charge_entry *e = l->entry;
            e->launching--;
            graph_submitted[e->dev] = 1;
            if (l->freed == e->freed) {
                e->launched = 1;
                if (!s->upload) e->generation++;
            } else {
                s->launched = 0;
            }
            graph_entry_update(e);
        }
        if (!s->fast) {
            for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
                if (s->temporary[dev] != 0) graph_submitted[dev] = 1;
            }
        }
    } else {
        graph_cancel_entries(s);
    }
    if (result == CUDA_SUCCESS && (graph_has_temporary(s) || s->has_frees)) {
        if (!s->context_dead) {
            CUevent event = s->prepared_event != NULL ? s->prepared_event : s->done;
            CUresult recorded = CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventRecord, event, stream);
            if (s->prepared_event != NULL) {
                /* Avoid destroying an event handle reused by the new context. */
                if (s->done != NULL && s->done != s->prepared_event)
                    CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->done);
                s->done = s->prepared_event;
                s->prepared_event = NULL;
                s->event_ctx = s->key.ctx;
                s->event_ctx_id = s->key.ctx_id;
            }
            s->recorded = recorded == CUDA_SUCCESS;
            if (!s->recorded) LOG_WARN("Cannot record graph completion; keeping its memory charge");
            if (!s->upload && s->has_frees) {
                CUevent free_event = s->prepared_free_event != NULL ? s->prepared_free_event : s->free_event;
                s->free_recorded = CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventRecord,
                    free_event, stream) == CUDA_SUCCESS;
                if (s->prepared_free_event != NULL) {
                    if (s->free_event != NULL && s->free_event != s->prepared_free_event)
                        CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->free_event);
                    s->free_event = s->prepared_free_event;
                    s->prepared_free_event = NULL;
                    s->free_event_ctx = s->key.ctx;
                    s->free_event_ctx_id = s->key.ctx_id;
                }
                s->free_key = s->key;
                s->free_submitted = s->submitted;
            }
        } else {
            graph_active_remove(s);
        }
    } else if (result != CUDA_SUCCESS) {
        if (!s->was_active || (s->context_dead && s->done == NULL)) graph_active_remove(s);
        s->key = s->saved_key;
        graph_active_dirty = 1;
        if (s->prepared_event != NULL) {
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->prepared_event);
            s->prepared_event = NULL;
        }
        if (s->prepared_free_event != NULL) {
            CUDA_OVERRIDE_CALL(cuda_library_entry, cuEventDestroy_v2, s->prepared_free_event);
            s->prepared_free_event = NULL;
        }
    }
    s->pending = 0;
    if (!s->fast) {
        graph_refresh_pool(0);
        graph_rebalance(0);
    }
    pthread_mutex_unlock(&graph_mutex);
    graph_unlock_entries(s);
    pthread_mutex_unlock(&s->submit_mutex);
    graph_exec_put_user(s);
}

/** Return the native context generation when the driver supports it. */
cuuint64_t graph_context_id(CUcontext ctx) {
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuCtxGetId) == NULL) return 0;
    cuuint64_t id;
    if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetId, ctx, &id) != CUDA_SUCCESS)
        return UINT64_MAX;
    return id;
}

/** Account confirmed external frees before context teardown invalidates their events. */
void graph_context_prepare_destroy(CUcontext ctx, cuuint64_t id) {
    if (ctx == NULL) return;
    pthread_mutex_lock(&graph_mutex);
    for (graph_exec_state *s = graph_states; s != NULL; s = s->next) {
        if (!s->pending && s->free_event_ctx == ctx && s->free_event_ctx_id == id)
            graph_complete_frees(s, 0);
    }
    graph_rebalance(0);
    pthread_mutex_unlock(&graph_mutex);
}

/** Retire charges and tracking associated with the destroyed context generation. */
void graph_context_destroyed(CUcontext ctx, cuuint64_t id) {
    if (ctx == NULL) return;
    pthread_mutex_lock(&graph_mutex);
    for (capture_link **pp = &capture_links; *pp != NULL;) {
        capture_link *l = *pp;
        if (l->ctx == ctx && l->ctx_id == id) {
            *pp = l->next;
            free(l);
        } else {
            pp = &l->next;
        }
    }
    for (graph_exec_state *s = graph_states, *next; s != NULL; s = next) {
        next = s->next;
        if (s->event_ctx == ctx && s->event_ctx_id == id) {
            s->done = NULL;
            s->event_ctx = NULL;
            s->recorded = 0;
            if (!s->pending) graph_active_remove(s);
        }
        if (s->free_event_ctx == ctx && s->free_event_ctx_id == id) {
            s->free_event = NULL;
            s->free_event_ctx = NULL;
            s->free_recorded = 0;
        }
        if (s->key.ctx == ctx && s->key.ctx_id == id) {
            s->context_dead = 1;
            s->prepared_event = NULL;
            s->prepared_free_event = NULL;
            if (!s->pending) graph_active_remove(s);
        }
        graph_exec_maybe_free(s);
    }
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        if (graph_primary_ctx[dev] == ctx && graph_primary_ctx_id[dev] == id)
            graph_primary_ctx[dev] = NULL;
    }
    graph_refresh();
    graph_rebalance(0);
    pthread_mutex_unlock(&graph_mutex);
}

/** Remember the primary context and its generation for a device. */
void graph_primary_context(CUdevice dev, CUcontext ctx) {
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) return;
    cuuint64_t id = graph_context_id(ctx);
    pthread_mutex_lock(&graph_mutex);
    graph_primary_ctx[dev] = ctx;
    graph_primary_ctx_id[dev] = id;
    pthread_mutex_unlock(&graph_mutex);
}

/** Preserve completed frees before primary-context release or reset. */
void graph_primary_context_prepare_release(CUdevice dev) {
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) return;
    pthread_mutex_lock(&graph_mutex);
    CUcontext ctx = graph_primary_ctx[dev];
    cuuint64_t id = graph_primary_ctx_id[dev];
    pthread_mutex_unlock(&graph_mutex);
    graph_context_prepare_destroy(ctx, id);
}

/** Retire primary-context accounting after release or reset. */
void graph_primary_context_released(CUdevice dev, int reset) {
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) return;
    pthread_mutex_lock(&graph_mutex);
    CUcontext ctx = graph_primary_ctx[dev];
    cuuint64_t id = graph_primary_ctx_id[dev];
    pthread_mutex_unlock(&graph_mutex);
    graph_context_destroyed(ctx, id);
    if (reset && ctx != NULL) graph_primary_context(dev, ctx);
}

/** Serialize stream destruction and identity retirement against graph submission lookup. */
CUresult graph_stream_destroy(CUstream stream) {
    pthread_mutex_lock(&graph_mutex);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamDestroy_v2, stream);
    if (res == CUDA_SUCCESS && stream != NULL && stream != CU_STREAM_LEGACY &&
        stream != CU_STREAM_PER_THREAD) {
        uint64_t epoch = ++graph_stream_epoch;
        for (graph_exec_state *s = graph_active; s != NULL; s = s->active_next) {
            if (!s->key.native_id && s->key.stream == stream && s->key.epoch == 0) {
                s->key.epoch = epoch;
                graph_active_dirty = 1;
            }
            if (s->pending && !s->saved_key.native_id && s->saved_key.stream == stream &&
                s->saved_key.epoch == 0) s->saved_key.epoch = epoch;
            if (!s->free_key.native_id && s->free_key.stream == stream && s->free_key.epoch == 0)
                s->free_key.epoch = epoch;
        }
        graph_rebalance(0);
    }
    pthread_mutex_unlock(&graph_mutex);
    return res;
}

uint64_t graph_sync_begin(void) {
    pthread_mutex_lock(&graph_mutex);
    uint64_t submitted = graph_submit_serial;
    pthread_mutex_unlock(&graph_mutex);
    return submitted;
}

/** Release completed graph charges covered by the synchronization snapshot. */
void graph_synchronized(CUcontext ctx, CUstream stream, int whole_context,
                        uint64_t submitted) {
    pthread_mutex_lock(&graph_mutex);
    graph_stream_key key = {0};
    cuuint64_t id = whole_context ? graph_context_id(ctx) : 0;
    if (!whole_context && graph_stream_info(stream, &key) != CUDA_SUCCESS) {
        pthread_mutex_unlock(&graph_mutex);
        return;
    }
    for (graph_exec_state *s = graph_active, *next; s != NULL; s = next) {
        next = s->active_next;
        graph_stream_key target = s->key;
        target.known = 1;
        int matches = whole_context ? s->key.ctx == ctx && s->key.ctx_id == id
                                    : graph_stream_equal(&target, &key);
        /* Preserve submissions newer than the synchronization snapshot. */
        if (!s->pending) {
            int frees_covered = s->free_submitted <= submitted && (whole_context ?
                s->free_key.ctx == ctx && s->free_key.ctx_id == id : graph_stream_equal(&s->free_key, &key));
            int frees_done = graph_complete_frees(s, frees_covered);
            if (matches && s->submitted <= submitted && frees_done) graph_active_remove(s);
        }
        graph_exec_maybe_free(s);
    }
    graph_refresh();
    graph_rebalance(0);
    pthread_mutex_unlock(&graph_mutex);
}

/** Release invalidated allocations across all participating streams. */
void graph_capture_end(cuuint64_t capid, CUgraph graph) {
    pthread_mutex_lock(&graph_mutex);
    for (graph_charge_entry *g = graph_charges, *n; g != NULL; g = n) {
        n = g->next;
        if (!g->capturing) continue;
        if (graph != NULL && g->graph == graph) {
            g->capturing = 0;  /* ended well, now owned by the graph */
        } else if (graph == NULL && capid != 0 && g->capid == capid) {
            g->capturing = 0;
            g->graph = NULL;
            g->launched = 0;
            g->freed++;
            graph_entry_update(g);
            if (g->nexec == 0 && g->launching == 0) graph_drop_entry(g);
        }
    }
    graph_refresh();
    graph_rebalance(0);
    pthread_mutex_unlock(&graph_mutex);
}

/** Retain persistent allocations until their pointers are freed. */
void graph_release(CUgraph graph) {
    pthread_mutex_lock(&graph_mutex);
    for (graph_charge_entry *g = graph_charges, *n; g != NULL; g = n) {
        n = g->next;
        if (g->graph != graph && g->group->owner != graph) continue;
        g->graph = NULL;
        /* Memory of a graph that never ran was never allocated. */
        if (!g->launched && g->nexec == 0) graph_drop_entry(g);
    }
    for (graph_group *group = graph_groups; group != NULL; group = group->next) {
        if (group->owner == graph) {
            group->graph = NULL;
            group->owner = NULL;
        }
    }
    graph_refresh();
    graph_rebalance(0);
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
    CUstreamCaptureStatus cap;
    cuuint64_t capid;
    CUgraph capgraph;
    graph_capture_info(hStream, &cap, &capid, &capgraph);
    if (cap != CU_STREAM_CAPTURE_STATUS_NONE) {
        if (capgraph != NULL) {
            if (graph_record(capgraph, e->entry->dev, size, e->entry->address, capid) != 0) {
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

/** Bind a previously unknown capture ID before recording its first allocation. */
static CUresult graph_capture_observe(CUstream stream, cuuint64_t id, CUgraph graph) {
    cuuint64_t cached;
    uint64_t token;
    CUresult res = graph_capture_resolve(stream, &cached, &token);
    if (res != CUDA_SUCCESS || token == 0 || cached != 0) return res;
    pthread_mutex_lock(&graph_mutex);
    for (capture_link *l = capture_links; l != NULL; l = l->next) {
        if (l->token != token) continue;
        l->id = id;
        l->graph = graph;
        break;
    }
    pthread_mutex_unlock(&graph_mutex);
    return CUDA_SUCCESS;
}

/** Admit captured allocations before the driver records their nodes. */
static CUresult allocate_captured(CUdeviceptr *address, size_t size, CUdevice dev,
                                  CUmemoryPool pool, CUstream stream, int from_pool, int *capturing) {
    CUstreamCaptureStatus cap;
    cuuint64_t id;
    CUgraph graph;
    CUresult res = graph_capture_info(stream, &cap, &id, &graph);
    *capturing = res != CUDA_SUCCESS || cap != CU_STREAM_CAPTURE_STATUS_NONE;
    if (res != CUDA_SUCCESS || !*capturing) return res;
    if (graph == NULL) return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    res = graph_capture_observe(stream, id, graph);
    if (res != CUDA_SUCCESS) return res;
    graph_charge_entry *entry;
    res = graph_alloc_begin(graph, dev, size, id, 1, &entry);
    if (res != CUDA_SUCCESS) return res;
    res = from_pool
        ? CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemAllocFromPoolAsync, address, size, pool, stream)
        : CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemAllocAsync, address, size, stream);
    graph_alloc_end(entry, res == CUDA_SUCCESS ? *address : 0, res);
    return res;
}

/** Reserve captured allocation quota before native default-pool allocation. */
int add_chunk_async(CUdeviceptr *address, size_t size, CUstream hStream) {
    size_t addr = 0;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    int capturing;
    res = allocate_captured(address, size, dev, NULL, hStream, 0, &capturing);
    if (capturing) return res;
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

/** Reserve captured allocation quota before native caller-pool allocation. */
int add_chunk_from_pool_async(CUdeviceptr *address, size_t size, CUmemoryPool pool, CUstream hStream) {
    size_t addr = 0;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    int capturing;
    res = allocate_captured(address, size, dev, pool, hStream, 1, &capturing);
    if (capturing) return res;
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

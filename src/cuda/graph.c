#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/libcuda_hook.h"
#include "include/memory_limit.h"
#include "allocator/allocator.h"

extern int pidfound;
extern void rate_limiter(int64_t grids, int64_t blocks);

static void conditional_forget(CUgraph graph, CUgraphNode node);

CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags){
	LOG_DEBUG("cuGraphCreate");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphCreate,phGraph,flags);
}

CUresult cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphAddKernelNode_v2");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddKernelNode_v2,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
}

CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphKernelNodeGetParams_v2");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphKernelNodeGetParams_v2,hNode,nodeParams);
}

CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphKernelNodeSetParams_v2");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphKernelNodeSetParams_v2,hNode,nodeParams);
}

CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
	LOG_DEBUG("cuGraphAddMemcpyNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddMemcpyNode,phGraphNode,hGraph,dependencies,numDependencies,copyParams,ctx);
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams) {
	LOG_DEBUG("cuGraphMemcpyNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphMemcpyNodeGetParams,hNode,nodeParams);
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams) {
	LOG_DEBUG("cuGraphMemcpyNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphMemcpyNodeSetParams,hNode,nodeParams);
}

CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
	LOG_DEBUG("cuGraphAddMemsetNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddMemsetNode,phGraphNode,hGraph,dependencies,numDependencies,memsetParams,ctx);
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphMemsetNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphMemsetNodeGetParams,hNode,nodeParams);
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphMemsetNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphMemsetNodeSetParams,hNode,nodeParams);
}

CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphAddHostNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddHostNode,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
}

CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphHostNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphHostNodeGetParams,hNode,nodeParams);
}

CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphHostNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphHostNodeSetParams,hNode,nodeParams);
}

CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph) {
	LOG_DEBUG("cuGraphAddChildGraphNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddChildGraphNode,phGraphNode,hGraph,dependencies,numDependencies,childGraph);
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
	LOG_DEBUG("cuGraphChildGraphNodeGetGraph");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphChildGraphNodeGetGraph,hNode,phGraph);
}

CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies) {
	LOG_DEBUG("cuGraphAddEmptyNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddEmptyNode,phGraphNode,hGraph,dependencies,numDependencies);
}

CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
	LOG_DEBUG("cuGraphAddEventRecordNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddEventRecordNode,phGraphNode,hGraph,dependencies,numDependencies,event);
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
	LOG_DEBUG("cuGraphEventRecordNodeGetEvent");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphEventRecordNodeGetEvent,hNode,event_out);
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
	LOG_DEBUG("cuGraphEventRecordNodeSetEvent");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphEventRecordNodeSetEvent,hNode,event);
}

CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
	LOG_DEBUG("cuGraphAddEventWaitNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddEventWaitNode,phGraphNode,hGraph,dependencies,numDependencies,event);
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
	LOG_DEBUG("cuGraphEventWaitNodeGetEvent");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphEventWaitNodeGetEvent,hNode,event_out);
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
	LOG_DEBUG("cuGraphEventWaitNodeSetEvent");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphEventWaitNodeSetEvent,hNode,event);
}

CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphAddExternalSemaphoresSignalNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddExternalSemaphoresSignalNode,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
	LOG_DEBUG("cuGraphExternalSemaphoresSignalNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExternalSemaphoresSignalNodeGetParams,hNode,params_out);
}

CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphExternalSemaphoresSignalNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExternalSemaphoresSignalNodeSetParams,hNode,nodeParams);
}

CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphAddExternalSemaphoresWaitNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddExternalSemaphoresWaitNode,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
	LOG_DEBUG("cuGraphExternalSemaphoresWaitNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExternalSemaphoresWaitNodeGetParams,hNode,params_out);
}

CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphExternalSemaphoresWaitNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExternalSemaphoresWaitNodeSetParams,hNode,nodeParams);
}

CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphExecExternalSemaphoresSignalNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExecExternalSemaphoresSignalNodeSetParams,hGraphExec,hNode,nodeParams);
}

CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphExecExternalSemaphoresWaitNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExecExternalSemaphoresWaitNodeSetParams,hGraphExec,hNode,nodeParams);
}

CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
	LOG_DEBUG("cuGraphClone");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphClone,phGraphClone,originalGraph);
}

CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) {
	LOG_DEBUG("cuGraphNodeFindInClone");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeFindInClone,phNode,hOriginalNode,hClonedGraph);
}

CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
	LOG_DEBUG("cuGraphNodeGetType");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeGetType,hNode,type);
}

CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes){
	LOG_DEBUG("cuGraphGetNodes");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphGetNodes,hGraph,nodes,numNodes);
}

CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes) {
	LOG_DEBUG("cuGraphGetRootNodes");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphGetRootNodes,hGraph,rootNodes,numRootNodes);
}

#if CUDA_VERSION < 13000
CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges) {
	LOG_DEBUG("cuGraphGetEdges");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphGetEdges,hGraph,from,to,numEdges);
}
#endif

CUresult cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, CUgraphEdgeData *edgeData,
                            size_t *numEdges) {
    LOG_DEBUG("cuGraphGetEdges_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphGetEdges_v2, hGraph, from, to, edgeData, numEdges);
}

#if CUDA_VERSION < 13000
CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies) {
	LOG_DEBUG("cuGraphNodeGetDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeGetDependencies,hNode,dependencies,numDependencies);
}
#endif

CUresult cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode *dependencies, CUgraphEdgeData *edgeData,
                                       size_t *numDependencies) {
    LOG_DEBUG("cuGraphNodeGetDependencies_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphNodeGetDependencies_v2, hNode, dependencies, edgeData,
                              numDependencies);
}

#if CUDA_VERSION < 13000
CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes) {
	LOG_DEBUG("cuGraphNodeGetDependentNodes");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeGetDependentNodes,hNode,dependentNodes,numDependentNodes);
}
#endif

CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode, CUgraphNode *dependentNodes, CUgraphEdgeData *edgeData,
                                         size_t *numDependentNodes) {
    LOG_DEBUG("cuGraphNodeGetDependentNodes_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphNodeGetDependentNodes_v2, hNode, dependentNodes, edgeData,
                              numDependentNodes);
}

#if CUDA_VERSION < 13000
CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
	LOG_DEBUG("cuGraphAddDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddDependencies,hGraph,from,to,numDependencies);
}
#endif

CUresult cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to,
                                   const CUgraphEdgeData *edgeData, size_t numDependencies) {
    LOG_DEBUG("cuGraphAddDependencies_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphAddDependencies_v2, hGraph, from, to, edgeData,
                              numDependencies);
}

#if CUDA_VERSION < 13000
CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
	LOG_DEBUG("cuGraphRemoveDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphRemoveDependencies,hGraph,from,to,numDependencies);
}
#endif

CUresult cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to,
                                      const CUgraphEdgeData *edgeData, size_t numDependencies) {
    LOG_DEBUG("cuGraphRemoveDependencies_v2");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphRemoveDependencies_v2, hGraph, from, to, edgeData,
                              numDependencies);
}

CUresult cuGraphDestroyNode(CUgraphNode hNode) {
    LOG_DEBUG("cuGraphDestroyNode");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphDestroyNode, hNode);
    if (res == CUDA_SUCCESS) conditional_forget(NULL, hNode);
    return res;
}

/* A graph replay never passes through cuLaunchKernel, so its kernels would skip
 * rate_limiter and a graph-heavy process could run far above CUDA_DEVICE_SM_LIMIT.
 * Each executable graph is charged, at every launch, the summed threads
 * (grid x block) of the kernel nodes it was instantiated from: the unit
 * cuLaunchKernel is charged per kernel. Costs live in an open-addressing table
 * keyed by CUgraphExec; deletion shifts entries back so lookups stay short.
 * The cost of each top-level kernel and child-graph node is kept too, because
 * cuGraphExec*NodeSetParams changes one node of the exec without a new graph. */
#define LAUNCH_COST_SLOTS 65536  /* power of two */
#define LAUNCH_COST_MAX_DEPTH 8  /* child-graph nesting followed when summing */

typedef struct {
    CUgraphNode node;
    int64_t threads;
    size_t seq;  /* position among the listed nodes in cuGraphGetNodes order */
} node_cost;

typedef struct {
    node_cost *items;  /* sorted by node */
    size_t len, cap;
    int64_t rest;        /* conditional bodies */
} graph_cost;

typedef struct {
    CUgraphExec exec;
    int64_t threads;
    graph_cost cost;
} launch_cost_slot;

static launch_cost_slot launch_costs[LAUNCH_COST_SLOTS];
static pthread_mutex_t launch_cost_mutex = PTHREAD_MUTEX_INITIALIZER;
static size_t launch_cost_used;
static int launch_cost_full_logged;

static size_t launch_cost_home(CUgraphExec exec) {
    uint64_t x = (uint64_t)(uintptr_t)exec;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return (size_t)(x & (LAUNCH_COST_SLOTS - 1));
}

static int64_t saturating_add(int64_t a, int64_t b) {
    return a > INT64_MAX - b ? INT64_MAX : a + b;
}

static int64_t kernel_threads(unsigned int gx, unsigned int gy, unsigned int gz,
                              unsigned int bx, unsigned int by, unsigned int bz) {
    int64_t grid = (int64_t)gx * gy * gz;
    int64_t block = (int64_t)bx * by * bz;
    return (block != 0 && grid > INT64_MAX / block) ? INT64_MAX : grid * block;
}

static int64_t graph_cost_total(const graph_cost *c) {
    int64_t total = c->rest;
    for (size_t i = 0; i < c->len; i++) total = saturating_add(total, c->items[i].threads);
    return total;
}

static int node_cost_cmp(const void *a, const void *b) {
    uintptr_t x = (uintptr_t)((const node_cost *)a)->node, y = (uintptr_t)((const node_cost *)b)->node;
    return x < y ? -1 : x > y;
}

/** Keep a node's cost; without memory the node is folded into rest and cannot be changed later. */
static void graph_cost_add(graph_cost *c, CUgraphNode node, int64_t threads) {
    if (c->len == c->cap) {
        size_t grown = c->cap ? c->cap * 2 : 16;
        node_cost *items = realloc(c->items, grown * sizeof(*items));
        if (items == NULL) {
            c->rest = saturating_add(c->rest, threads);
            return;
        }
        c->items = items;
        c->cap = grown;
    }
    c->items[c->len].node = node;
    c->items[c->len].threads = threads;
    c->items[c->len].seq = c->len;
    c->len++;
}

/* Body graphs of conditional nodes. The driver hands them out only when the node is
 * created (cuGraphAddNode), so they are remembered here for launch costs. */
typedef struct conditional_body {
    CUgraphNode node;
    CUgraph parent;
    CUgraph *bodies;
    unsigned int size;
    struct conditional_body *next;
} conditional_body;

static conditional_body *conditional_bodies;
static pthread_mutex_t conditional_mutex = PTHREAD_MUTEX_INITIALIZER;

/** Remember the body graphs of a conditional node created in parent. */
void graph_conditional_added(CUgraphNode node, CUgraph parent, const CUgraph *bodies, unsigned int size) {
    if (node == NULL || bodies == NULL || size == 0) return;
    conditional_body *c = calloc(1, sizeof(*c));
    if (c == NULL) return;
    c->bodies = malloc(size * sizeof(*c->bodies));
    if (c->bodies == NULL) {
        free(c);
        return;
    }
    memcpy(c->bodies, bodies, size * sizeof(*c->bodies));
    c->node = node;
    c->parent = parent;
    c->size = size;
    pthread_mutex_lock(&conditional_mutex);
    c->next = conditional_bodies;
    conditional_bodies = c;
    pthread_mutex_unlock(&conditional_mutex);
}

/** Unlink c, queue its body graphs (their own conditionals go too) and free it. */
static void conditional_unlink(conditional_body **p, CUgraph **queue, size_t *len, size_t *cap) {
    conditional_body *c = *p;
    *p = c->next;
    for (unsigned int i = 0; i < c->size; i++) {
        if (*len == *cap) {
            size_t grown = *cap ? *cap * 2 : 16;
            CUgraph *q = realloc(*queue, grown * sizeof(**queue));
            if (q == NULL) break;
            *queue = q;
            *cap = grown;
        }
        (*queue)[(*len)++] = c->bodies[i];
    }
    free(c->bodies);
    free(c);
}

/** Forget the conditional nodes of a destroyed graph, or one destroyed node, and everything nested. */
static void conditional_forget(CUgraph graph, CUgraphNode node) {
    CUgraph *queue = NULL;
    size_t len = 0, cap = 0;
    pthread_mutex_lock(&conditional_mutex);
    for (conditional_body **p = &conditional_bodies; *p != NULL;) {
        if ((graph != NULL && (*p)->parent == graph) || (node != NULL && (*p)->node == node))
            conditional_unlink(p, &queue, &len, &cap);
        else
            p = &(*p)->next;
    }
    while (len > 0) {
        CUgraph body = queue[--len];
        for (conditional_body **p = &conditional_bodies; *p != NULL;) {
            if ((*p)->parent == body)
                conditional_unlink(p, &queue, &len, &cap);
            else
                p = &(*p)->next;
        }
    }
    pthread_mutex_unlock(&conditional_mutex);
    free(queue);
}

/** Largest launch cost among a conditional node's bodies; IF and SWITCH run at most one body.
 * A WHILE node reruns its body until the condition is false. The number of runs is not
 * known at launch, so only one is charged and loops are undercharged. */
static int64_t graph_kernel_threads(CUgraph graph, int depth, graph_cost *top);
static int64_t conditional_threads(CUgraphNode node, int depth) {
    CUgraph *bodies = NULL;
    unsigned int size = 0;
    pthread_mutex_lock(&conditional_mutex);
    for (conditional_body *c = conditional_bodies; c != NULL; c = c->next) {
        if (c->node != node) continue;
        bodies = malloc(c->size * sizeof(*bodies));
        if (bodies != NULL) {
            memcpy(bodies, c->bodies, c->size * sizeof(*bodies));
            size = c->size;
        }
        break;
    }
    pthread_mutex_unlock(&conditional_mutex);
    int64_t most = 0;
    for (unsigned int i = 0; i < size; i++) {
        int64_t threads = graph_kernel_threads(bodies[i], depth + 1, NULL);
        if (threads > most) most = threads;
    }
    free(bodies);
    return most;
}

/** Sum grid x block over the kernel nodes of a graph and its child graphs.
 * With top, the graph's own kernel and child-graph nodes are listed there one by one and the rest summed. */
static int64_t graph_kernel_threads(CUgraph graph, int depth, graph_cost *top) {
    size_t count = 0;
    if (depth > LAUNCH_COST_MAX_DEPTH) return 0;
    /* A driver without the node queries leaves the graph uncharged, as before. */
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuGraphGetNodes) == NULL ||
        CUDA_FIND_ENTRY(cuda_library_entry, cuGraphNodeGetType) == NULL ||
        CUDA_FIND_ENTRY(cuda_library_entry, cuGraphKernelNodeGetParams_v2) == NULL ||
        CUDA_FIND_ENTRY(cuda_library_entry, cuGraphChildGraphNodeGetGraph) == NULL)
        return 0;
    if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphGetNodes, graph, NULL, &count) != CUDA_SUCCESS)
        return 0;
    if (count == 0) return 0;
    CUgraphNode *nodes = malloc(count * sizeof(*nodes));
    if (nodes == NULL) return 0;
    int64_t total = 0;
    if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphGetNodes, graph, nodes, &count) == CUDA_SUCCESS) {
        for (size_t i = 0; i < count; i++) {
            CUgraphNodeType type;
            if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphNodeGetType, nodes[i], &type) != CUDA_SUCCESS)
                continue;
            if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
                CUDA_KERNEL_NODE_PARAMS p;
                if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphKernelNodeGetParams_v2, nodes[i], &p) !=
                    CUDA_SUCCESS)
                    continue;
                int64_t threads = kernel_threads(p.gridDimX, p.gridDimY, p.gridDimZ,
                                                 p.blockDimX, p.blockDimY, p.blockDimZ);
                total = saturating_add(total, threads);
                if (top != NULL) graph_cost_add(top, nodes[i], threads);
                continue;
#if CUDA_VERSION >= 12030
            } else if (type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
                int64_t threads = conditional_threads(nodes[i], depth);
                total = saturating_add(total, threads);
                if (top != NULL) top->rest = saturating_add(top->rest, threads);
#endif
            } else if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
                CUgraph child;
                if (CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphChildGraphNodeGetGraph, nodes[i], &child) ==
                    CUDA_SUCCESS) {
                    int64_t threads = graph_kernel_threads(child, depth + 1, NULL);
                    total = saturating_add(total, threads);
                    if (top != NULL) graph_cost_add(top, nodes[i], threads);
                }
            }
        }
    }
    free(nodes);
    if (top != NULL && top->len > 1) qsort(top->items, top->len, sizeof(*top->items), node_cost_cmp);
    return total;
}

/** Remember the launch cost of an executable graph, taking ownership of cost; zero is not stored. */
static void launch_cost_put(CUgraphExec exec, int64_t threads, graph_cost cost) {
    if (exec == NULL || threads <= 0) {
        free(cost.items);
        return;
    }
    pthread_mutex_lock(&launch_cost_mutex);
    size_t i = launch_cost_home(exec);
    for (size_t n = 0; n < LAUNCH_COST_SLOTS; n++, i = (i + 1) & (LAUNCH_COST_SLOTS - 1)) {
        /* Keep one slot empty so every probe, including the shift in launch_cost_drop, ends. */
        if (launch_costs[i].exec == NULL && launch_cost_used >= LAUNCH_COST_SLOTS - 1) break;
        if (launch_costs[i].exec == NULL || launch_costs[i].exec == exec) {
            if (launch_costs[i].exec == NULL) launch_cost_used++;
            free(launch_costs[i].cost.items);
            launch_costs[i].exec = exec;
            launch_costs[i].threads = threads;
            launch_costs[i].cost = cost;
            pthread_mutex_unlock(&launch_cost_mutex);
            return;
        }
    }
    if (!launch_cost_full_logged) {
        launch_cost_full_logged = 1;
        LOG_WARN("graph launch cost table full: graphs instantiated from now on are not rate limited");
    }
    pthread_mutex_unlock(&launch_cost_mutex);
    free(cost.items);
}

/** Launch cost of an executable graph, or 0 when none was recorded. */
static int64_t launch_cost_get(CUgraphExec exec) {
    int64_t threads = 0;
    pthread_mutex_lock(&launch_cost_mutex);
    size_t i = launch_cost_home(exec);
    for (size_t n = 0; n < LAUNCH_COST_SLOTS && launch_costs[i].exec != NULL;
         n++, i = (i + 1) & (LAUNCH_COST_SLOTS - 1)) {
        if (launch_costs[i].exec == exec) {
            threads = launch_costs[i].threads;
            break;
        }
    }
    pthread_mutex_unlock(&launch_cost_mutex);
    return threads;
}

/** Forget an executable graph, shifting later entries back to keep probe chains intact. */
static void launch_cost_drop(CUgraphExec exec) {
    pthread_mutex_lock(&launch_cost_mutex);
    size_t i = launch_cost_home(exec);
    size_t n = 0;
    for (; n < LAUNCH_COST_SLOTS && launch_costs[i].exec != NULL; n++, i = (i + 1) & (LAUNCH_COST_SLOTS - 1)) {
        if (launch_costs[i].exec == exec) break;
    }
    if (n == LAUNCH_COST_SLOTS || launch_costs[i].exec == NULL) {
        pthread_mutex_unlock(&launch_cost_mutex);
        return;
    }
    free(launch_costs[i].cost.items);
    size_t hole = i;
    for (size_t j = (hole + 1) & (LAUNCH_COST_SLOTS - 1); launch_costs[j].exec != NULL;
         j = (j + 1) & (LAUNCH_COST_SLOTS - 1)) {
        size_t home = launch_cost_home(launch_costs[j].exec);
        /* Move j into the hole unless its home lies cyclically in (hole, j]. */
        if (((j - home) & (LAUNCH_COST_SLOTS - 1)) >= ((j - hole) & (LAUNCH_COST_SLOTS - 1))) {
            launch_costs[hole] = launch_costs[j];
            hole = j;
        }
    }
    launch_costs[hole].exec = NULL;
    launch_costs[hole].threads = 0;
    launch_costs[hole].cost = (graph_cost){0};
    launch_cost_used--;
    pthread_mutex_unlock(&launch_cost_mutex);
}

/** After cuGraphExecUpdate, key the new costs by the nodes exec was instantiated with.
 * cuGraphExec*NodeSetParams keeps taking those nodes, not the ones of the update graph.
 * Nodes are paired by their position in cuGraphGetNodes, which matches the driver when both
 * graphs add their nodes in the same order. The driver pairs by edges, so if the order
 * differs the total stays right, but a later setter may replace another node's cost. */
static void launch_cost_keep_nodes(CUgraphExec exec, graph_cost *cost) {
    pthread_mutex_lock(&launch_cost_mutex);
    size_t i = launch_cost_home(exec);
    for (size_t n = 0; n < LAUNCH_COST_SLOTS && launch_costs[i].exec != NULL;
         n++, i = (i + 1) & (LAUNCH_COST_SLOTS - 1)) {
        if (launch_costs[i].exec != exec) continue;
        const graph_cost *old = &launch_costs[i].cost;
        CUgraphNode *keys = old->len == cost->len && cost->len > 0 ? malloc(cost->len * sizeof(*keys)) : NULL;
        if (keys != NULL) {
            for (size_t j = 0; j < old->len; j++) keys[old->items[j].seq] = old->items[j].node;
            for (size_t j = 0; j < cost->len; j++) cost->items[j].node = keys[cost->items[j].seq];
            free(keys);
            qsort(cost->items, cost->len, sizeof(*cost->items), node_cost_cmp);
        }
        break;
    }
    pthread_mutex_unlock(&launch_cost_mutex);
}

/** Store a new launch cost for exec, or forget it when the graph has no kernels. */
static void launch_cost_record(CUgraphExec exec, CUgraph graph, int updated) {
    graph_cost cost = {0};
    int64_t threads = graph_kernel_threads(graph, 0, &cost);
    if (updated) launch_cost_keep_nodes(exec, &cost);
    if (threads > 0) {
        launch_cost_put(exec, threads, cost);
    } else {
        free(cost.items);
        launch_cost_drop(exec);
    }
}

/** Charge one kernel or child-graph node of exec its new launch size; nodes exec was not built with keep their cost. */
static void launch_cost_set_node(CUgraphExec exec, CUgraphNode node, int64_t threads) {
    pthread_mutex_lock(&launch_cost_mutex);
    size_t i = launch_cost_home(exec);
    for (size_t n = 0; n < LAUNCH_COST_SLOTS && launch_costs[i].exec != NULL;
         n++, i = (i + 1) & (LAUNCH_COST_SLOTS - 1)) {
        if (launch_costs[i].exec != exec) continue;
        graph_cost *c = &launch_costs[i].cost;
        node_cost key = {.node = node};
        node_cost *k = c->len ? bsearch(&key, c->items, c->len, sizeof(*c->items), node_cost_cmp) : NULL;
        if (k != NULL) {
            k->threads = threads;
            launch_costs[i].threads = graph_cost_total(c);
        }
        break;
    }
    pthread_mutex_unlock(&launch_cost_mutex);
}

/** Commit executable-graph tracking on success or discard it after native failure. */
static CUresult track_exec(CUresult res, graph_exec_state *state, CUgraph graph, CUgraphExec *phGraphExec) {
    if (res == CUDA_SUCCESS) {
        graph_exec_commit(state, *phGraphExec);
        launch_cost_record(*phGraphExec, graph, 0);
    } else {
        graph_exec_discard(state);
    }
    return res;
}

/** Prepare graph-accounting state before native graph instantiation. */
CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode,
                            char *logBuffer, size_t bufferSize) {
    LOG_DEBUG("cuGraphInstantiate");
    graph_exec_state *state;
    CUresult res = graph_exec_prepare(hGraph, &state);
    if (res != CUDA_SUCCESS) return res;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphInstantiate, phGraphExec, hGraph,
                                      phErrorNode, logBuffer, bufferSize);
    return track_exec(res, state, hGraph, phGraphExec);
}

/** Prepare graph-accounting state before instantiating a graph with flags. */
CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags) {
    LOG_DEBUG("cuGraphInstantiateWithFlags");
    graph_exec_state *state;
    CUresult res = graph_exec_prepare(hGraph, &state);
    if (res != CUDA_SUCCESS) return res;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphInstantiateWithFlags, phGraphExec,
                                      hGraph, flags);
    return track_exec(res, state, hGraph, phGraphExec);
}

/** Instantiate a graph with parameters and reserve quota for any requested upload. */
static CUresult instantiate_with_params(cuda_sym_t instantiate, CUgraphExec *exec, CUgraph graph,
                                        CUDA_GRAPH_INSTANTIATE_PARAMS *params, int per_thread) {
    if (instantiate == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    graph_exec_state *state;
    CUresult res = graph_exec_prepare(graph, &state);
    if (res != CUDA_SUCCESS) return res;
    int upload = params != NULL && (params->flags & CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD);
    CUstream stream = upload ? params->hUploadStream : NULL;
    if (per_thread && stream == NULL) stream = CU_STREAM_PER_THREAD;
    if (upload && (res = graph_exec_reserve(state, stream, 1)) != CUDA_SUCCESS) {
        graph_exec_discard(state);
        return res;
    }
    res = instantiate(exec, graph, params);
    if (upload) graph_exec_end(state, stream, res);
    return track_exec(res, state, graph, exec);
}

CUresult cuGraphInstantiateWithParams(CUgraphExec *exec, CUgraph graph,
                                      CUDA_GRAPH_INSTANTIATE_PARAMS *params) {
    return instantiate_with_params((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry,
        cuGraphInstantiateWithParams), exec, graph, params, 0);
}


/** Reserve graph quota before native launch or upload and finalize the submission. */
static CUresult submit_graph(cuda_sym_t submit, CUgraphExec exec, CUstream stream, int per_thread, int upload) {
    graph_exec_state *state;
    CUstream accounting_stream = per_thread && stream == NULL ? CU_STREAM_PER_THREAD : stream;
    if (!upload) {
        /* Same admission as cuLaunchKernel, charged for the whole graph at once. rate_limiter
         * uses the current device, so a graph on several devices is charged to that one only. */
        ENSURE_RUNNING();
        pre_launch_kernel();
        if (pidfound == 1) {
            int64_t threads = launch_cost_get(exec);
            if (threads > 0) rate_limiter(threads, 1);
        }
    }
    CUresult res = graph_exec_begin(exec, accounting_stream, upload, &state);
    if (res != CUDA_SUCCESS) return res;
    res = submit(exec, stream);
    graph_exec_end(state, accounting_stream, res);
    return res;
}

CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    return submit_graph((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry, cuGraphUpload), hGraphExec, hStream, 0, 1);
}


CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    return submit_graph((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry, cuGraphLaunch), hGraphExec, hStream, 0, 0);
}

/** Preserve the original one-argument capture ABI while protecting allocation admission. */
CUresult cuStreamBeginCapture(CUstream stream) {
    uint64_t token;
    pthread_mutex_lock(&mutex);
    CUresult res = graph_capture_prepare(stream, CU_STREAM_CAPTURE_MODE_GLOBAL, &token);
    if (res == CUDA_SUCCESS) {
        res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuStreamBeginCapture, stream);
        graph_capture_started(token, res);
    }
    pthread_mutex_unlock(&mutex);
    return res;
}

/** Begin an original-ABI capture with per-thread default-stream semantics. */
CUresult cuStreamBeginCapture_ptsz(CUstream stream) {
    return cuStreamBeginCapture(stream == NULL ? CU_STREAM_PER_THREAD : stream);
}


/** Register capture protection before starting the native stream capture. */
static CUresult begin_capture(cuda_sym_t begin, CUstream stream, CUstreamCaptureMode mode, int per_thread) {
    pthread_mutex_lock(&mutex);
    uint64_t token;
    CUstream accounting_stream = per_thread && stream == NULL ? CU_STREAM_PER_THREAD : stream;
    CUresult res = graph_capture_prepare(accounting_stream, mode, &token);
    if (res != CUDA_SUCCESS) {
        pthread_mutex_unlock(&mutex);
        return res;
    }
    res = begin(stream, mode);
    graph_capture_started(token, res);
    pthread_mutex_unlock(&mutex);
    return res;
}

CUresult cuStreamBeginCapture_v2(CUstream stream, CUstreamCaptureMode mode) {
    return begin_capture((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry, cuStreamBeginCapture_v2), stream, mode, 0);
}


/** Release capture protection and finalize charges once native capture terminates. */
static CUresult end_capture(cuda_sym_t end, CUstream stream, CUgraph *graph, int per_thread) {
    pthread_mutex_lock(&mutex);
    CUstream accounting_stream = per_thread && stream == NULL ? CU_STREAM_PER_THREAD : stream;
    cuuint64_t id;
    uint64_t token;
    CUresult lookup = graph_capture_resolve(accounting_stream, &id, &token);
    if (lookup != CUDA_SUCCESS) {
        pthread_mutex_unlock(&mutex);
        return lookup;
    }
    CUresult res = end(stream, graph);
    CUstreamCaptureStatus cap;
    cuuint64_t current_id;
    CUgraph current_graph;
    /* An invalidated capture terminates even if its follow-up status query fails. */
    int ended = res == CUDA_SUCCESS || res == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    if (!ended && graph_capture_info(accounting_stream, &cap, &current_id, &current_graph) == CUDA_SUCCESS)
        ended = cap == CU_STREAM_CAPTURE_STATUS_NONE || (id != 0 && current_id != 0 && id != current_id);
    if (ended) {
        graph_capture_finish(token);
        graph_capture_end(id, res == CUDA_SUCCESS && graph != NULL ? *graph : NULL);
    }
    pthread_mutex_unlock(&mutex);
    return res;
}

CUresult cuStreamEndCapture(CUstream stream, CUgraph *graph) {
    return end_capture((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry, cuStreamEndCapture), stream, graph, 0);
}


CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
	LOG_DEBUG("cuGraphExecDestroy");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphExecDestroy, hGraphExec);
    if (res == CUDA_SUCCESS) {
        graph_exec_destroyed(hGraphExec);
        launch_cost_drop(hGraphExec);
    }
    return res;
}

/** Recharge an executable graph after its topology or parameters were replaced. */
CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo *resultInfo) {
    LOG_DEBUG("cuGraphExecUpdate_v2");
    /* Drivers before 12.0 only export the legacy signature, which takes different arguments. */
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuGraphExecUpdate_v2) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphExecUpdate_v2, hGraphExec, hGraph, resultInfo);
    if (res == CUDA_SUCCESS) launch_cost_record(hGraphExec, hGraph, 1);
    return res;
}

/** Recharge an executable graph updated through the pre-12.0 signature. */
CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out,
                           CUgraphExecUpdateResult *updateResult_out) {
    LOG_DEBUG("cuGraphExecUpdate");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphExecUpdate, hGraphExec, hGraph, hErrorNode_out,
                                      updateResult_out);
    if (res == CUDA_SUCCESS) launch_cost_record(hGraphExec, hGraph, 1);
    return res;
}

/** Recharge one kernel node of an executable graph after its launch size changed. */
CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode,
                                           const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    LOG_DEBUG("cuGraphExecKernelNodeSetParams_v2");
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuGraphExecKernelNodeSetParams_v2) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphExecKernelNodeSetParams_v2, hGraphExec, hNode,
                                      nodeParams);
    if (res == CUDA_SUCCESS)
        launch_cost_set_node(hGraphExec, hNode, kernel_threads(nodeParams->gridDimX, nodeParams->gridDimY,
            nodeParams->gridDimZ, nodeParams->blockDimX, nodeParams->blockDimY, nodeParams->blockDimZ));
    return res;
}

/** Recharge one kernel node through the pre-12.0 signature. */
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                        const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams) {
    LOG_DEBUG("cuGraphExecKernelNodeSetParams");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphExecKernelNodeSetParams, hGraphExec, hNode,
                                      nodeParams);
    if (res == CUDA_SUCCESS)
        launch_cost_set_node(hGraphExec, hNode, kernel_threads(nodeParams->gridDimX, nodeParams->gridDimY,
            nodeParams->gridDimZ, nodeParams->blockDimX, nodeParams->blockDimY, nodeParams->blockDimZ));
    return res;
}

/** Recharge a kernel or child-graph node changed through the generic node setter. */
CUresult cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams *nodeParams) {
    LOG_DEBUG("cuGraphExecNodeSetParams");
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuGraphExecNodeSetParams) == NULL) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphExecNodeSetParams, hGraphExec, hNode, nodeParams);
    if (res != CUDA_SUCCESS) return res;
    if (nodeParams->type == CU_GRAPH_NODE_TYPE_KERNEL) {
        const CUDA_KERNEL_NODE_PARAMS_v3 *k = &nodeParams->kernel;
        launch_cost_set_node(hGraphExec, hNode, kernel_threads(k->gridDimX, k->gridDimY, k->gridDimZ,
                                                               k->blockDimX, k->blockDimY, k->blockDimZ));
    } else if (nodeParams->type == CU_GRAPH_NODE_TYPE_GRAPH) {
        launch_cost_set_node(hGraphExec, hNode, graph_kernel_threads(nodeParams->graph.graph, 1, NULL));
    }
    return res;
}

/** Recharge a child-graph node whose graph was replaced. */
CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) {
    LOG_DEBUG("cuGraphExecChildGraphNodeSetParams");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphExecChildGraphNodeSetParams, hGraphExec, hNode,
                                      childGraph);
    if (res == CUDA_SUCCESS) launch_cost_set_node(hGraphExec, hNode, graph_kernel_threads(childGraph, 1, NULL));
    return res;
}

CUresult cuGraphDestroy(CUgraph hGraph) {
    LOG_DEBUG("cuGraphDestroy");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphDestroy, hGraph);
    if (res == CUDA_SUCCESS) {
        graph_release(hGraph);
        conditional_forget(hGraph, NULL);
    }
    return res;
}

/** Trim supported native graph pools and refresh retained-pool accounting on success. */
CUresult cuDeviceGraphMemTrim(CUdevice device) {
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuDeviceGraphMemTrim) == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGraphMemTrim, device);
    if (res == CUDA_SUCCESS) graph_pool_trimmed();
    return res;
}

/** Launch through the accounting hook with per-thread NULL-stream semantics. */
CUresult cuGraphLaunch_ptsz(CUgraphExec exec, CUstream stream) {
    return cuGraphLaunch(exec, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}


/** Upload through the accounting hook with per-thread NULL-stream semantics. */
CUresult cuGraphUpload_ptsz(CUgraphExec exec, CUstream stream) {
    return cuGraphUpload(exec, stream == NULL ? CU_STREAM_PER_THREAD : stream);
}


/** Begin capture with per-thread NULL-stream semantics. */
CUresult cuStreamBeginCapture_v2_ptsz(CUstream stream, CUstreamCaptureMode mode) {
    return cuStreamBeginCapture_v2(stream == NULL ? CU_STREAM_PER_THREAD : stream, mode);
}


/** End capture with per-thread NULL-stream semantics. */
CUresult cuStreamEndCapture_ptsz(CUstream stream, CUgraph *graph) {
    return cuStreamEndCapture(stream == NULL ? CU_STREAM_PER_THREAD : stream, graph);
}


/** Reserve per-thread upload demand before native parameter-based instantiation. */
CUresult cuGraphInstantiateWithParams_ptsz(CUgraphExec *exec, CUgraph graph,
                                          CUDA_GRAPH_INSTANTIATE_PARAMS *params) {
    return instantiate_with_params((cuda_sym_t)CUDA_FIND_ENTRY(cuda_library_entry,
        cuGraphInstantiateWithParams_ptsz), exec, graph, params, 1);
}

#include "include/libcuda_hook.h"

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

CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
                                size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
        LOG_INFO("call in cuGraphAddMemAllocNode");
        return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddMemAllocNode,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
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

CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges) {
	LOG_DEBUG("cuGraphGetEdges");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphGetEdges,hGraph,from,to,numEdges);
}

CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies) {
	LOG_DEBUG("cuGraphNodeGetDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeGetDependencies,hNode,dependencies,numDependencies);
}

CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes) {
	LOG_DEBUG("cuGraphNodeGetDependentNodes");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeGetDependentNodes,hNode,dependentNodes,numDependentNodes);
}

CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
	LOG_DEBUG("cuGraphAddDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddDependencies,hGraph,from,to,numDependencies);
}

CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
	LOG_DEBUG("cuGraphRemoveDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphRemoveDependencies,hGraph,from,to,numDependencies);
}

CUresult cuGraphDestroyNode(CUgraphNode hNode) {
	LOG_DEBUG("cuGraphDestroyNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphDestroyNode,hNode);
}

CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize) {
	LOG_DEBUG("cuGraphInstantiate");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphInstantiate,phGraphExec,hGraph,phErrorNode,logBuffer,bufferSize);
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags) {
	LOG_DEBUG("cuGraphInstantiateWithFlags");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphInstantiateWithFlags,phGraphExec,hGraph,flags);
}

CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
	LOG_DEBUG("cuGraphUpload");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphUpload,hGraphExec,hStream);
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
	LOG_DEBUG("cuGraphLaunch");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphLaunch,hGraphExec,hStream);
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
	LOG_DEBUG("cuGraphExecDestroy");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExecDestroy,hGraphExec);
}

CUresult cuGraphDestroy(CUgraph hGraph) {
	LOG_DEBUG("cuGraphDestroy");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphDestroy,hGraph);
}
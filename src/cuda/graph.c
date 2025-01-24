#include "include/libcuda_hook.h"

CUresult hacked_cuGraphCreate(CUgraph *phGraph, unsigned int flags){
	LOG_DEBUG("cuGraphCreate");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphCreate,phGraph,flags);
}

CUresult hacked_cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphAddKernelNode_v2");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddKernelNode_v2,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
}

CUresult hacked_cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphKernelNodeGetParams_v2");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphKernelNodeGetParams_v2,hNode,nodeParams);
}

CUresult hacked_cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphKernelNodeSetParams_v2");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphKernelNodeSetParams_v2,hNode,nodeParams);
}

CUresult hacked_cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
	LOG_DEBUG("cuGraphAddMemcpyNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddMemcpyNode,phGraphNode,hGraph,dependencies,numDependencies,copyParams,ctx);
}

CUresult hacked_cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams) {
	LOG_DEBUG("cuGraphMemcpyNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphMemcpyNodeGetParams,hNode,nodeParams);
}

CUresult hacked_cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams) {
	LOG_DEBUG("cuGraphMemcpyNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphMemcpyNodeSetParams,hNode,nodeParams);
}

CUresult hacked_cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
	LOG_DEBUG("cuGraphAddMemsetNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddMemsetNode,phGraphNode,hGraph,dependencies,numDependencies,memsetParams,ctx);
}

CUresult hacked_cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphMemsetNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphMemsetNodeGetParams,hNode,nodeParams);
}

CUresult hacked_cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphMemsetNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphMemsetNodeSetParams,hNode,nodeParams);
}

CUresult hacked_cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphAddHostNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddHostNode,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
}

CUresult hacked_cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphHostNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphHostNodeGetParams,hNode,nodeParams);
}

CUresult hacked_cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphHostNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphHostNodeSetParams,hNode,nodeParams);
}

CUresult hacked_cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph) {
	LOG_DEBUG("cuGraphAddChildGraphNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddChildGraphNode,phGraphNode,hGraph,dependencies,numDependencies,childGraph);
}

CUresult hacked_cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
	LOG_DEBUG("cuGraphChildGraphNodeGetGraph");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphChildGraphNodeGetGraph,hNode,phGraph);
}

CUresult hacked_cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies) {
	LOG_DEBUG("cuGraphAddEmptyNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddEmptyNode,phGraphNode,hGraph,dependencies,numDependencies);
}

CUresult hacked_cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
	LOG_DEBUG("cuGraphAddEventRecordNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddEventRecordNode,phGraphNode,hGraph,dependencies,numDependencies,event);
}

CUresult hacked_cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
	LOG_DEBUG("cuGraphEventRecordNodeGetEvent");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphEventRecordNodeGetEvent,hNode,event_out);
}

CUresult hacked_cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
	LOG_DEBUG("cuGraphEventRecordNodeSetEvent");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphEventRecordNodeSetEvent,hNode,event);
}

CUresult hacked_cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event) {
	LOG_DEBUG("cuGraphAddEventWaitNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddEventWaitNode,phGraphNode,hGraph,dependencies,numDependencies,event);
}

CUresult hacked_cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
	LOG_DEBUG("cuGraphEventWaitNodeGetEvent");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphEventWaitNodeGetEvent,hNode,event_out);
}

CUresult hacked_cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
	LOG_DEBUG("cuGraphEventWaitNodeSetEvent");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphEventWaitNodeSetEvent,hNode,event);
}

CUresult hacked_cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphAddExternalSemaphoresSignalNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddExternalSemaphoresSignalNode,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
}

CUresult hacked_cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
	LOG_DEBUG("cuGraphExternalSemaphoresSignalNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExternalSemaphoresSignalNodeGetParams,hNode,params_out);
}

CUresult hacked_cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphExternalSemaphoresSignalNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExternalSemaphoresSignalNodeSetParams,hNode,nodeParams);
}

CUresult hacked_cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphAddExternalSemaphoresWaitNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddExternalSemaphoresWaitNode,phGraphNode,hGraph,dependencies,numDependencies,nodeParams);
}

CUresult hacked_cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
	LOG_DEBUG("cuGraphExternalSemaphoresWaitNodeGetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExternalSemaphoresWaitNodeGetParams,hNode,params_out);
}

CUresult hacked_cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphExternalSemaphoresWaitNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExternalSemaphoresWaitNodeSetParams,hNode,nodeParams);
}

CUresult hacked_cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphExecExternalSemaphoresSignalNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExecExternalSemaphoresSignalNodeSetParams,hGraphExec,hNode,nodeParams);
}

CUresult hacked_cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
	LOG_DEBUG("cuGraphExecExternalSemaphoresWaitNodeSetParams");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExecExternalSemaphoresWaitNodeSetParams,hGraphExec,hNode,nodeParams);
}

CUresult hacked_cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
	LOG_DEBUG("cuGraphClone");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphClone,phGraphClone,originalGraph);
}

CUresult hacked_cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) {
	LOG_DEBUG("cuGraphNodeFindInClone");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeFindInClone,phNode,hOriginalNode,hClonedGraph);
}

CUresult hacked_cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
	LOG_DEBUG("cuGraphNodeGetType");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeGetType,hNode,type);
}

CUresult hacked_cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes){
	LOG_DEBUG("cuGraphGetNodes");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphGetNodes,hGraph,nodes,numNodes);
}

CUresult hacked_cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes) {
	LOG_DEBUG("cuGraphGetRootNodes");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphGetRootNodes,hGraph,rootNodes,numRootNodes);
}

CUresult hacked_cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges) {
	LOG_DEBUG("cuGraphGetEdges");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphGetEdges,hGraph,from,to,numEdges);
}

CUresult hacked_cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies) {
	LOG_DEBUG("cuGraphNodeGetDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeGetDependencies,hNode,dependencies,numDependencies);
}

CUresult hacked_cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes) {
	LOG_DEBUG("cuGraphNodeGetDependentNodes");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphNodeGetDependentNodes,hNode,dependentNodes,numDependentNodes);
}

CUresult hacked_cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
	LOG_DEBUG("cuGraphAddDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphAddDependencies,hGraph,from,to,numDependencies);
}

CUresult hacked_cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
	LOG_DEBUG("cuGraphRemoveDependencies");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphRemoveDependencies,hGraph,from,to,numDependencies);
}

CUresult hacked_cuGraphDestroyNode(CUgraphNode hNode) {
	LOG_DEBUG("cuGraphDestroyNode");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphDestroyNode,hNode);
}

CUresult hacked_cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize) {
	LOG_DEBUG("cuGraphInstantiate");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphInstantiate,phGraphExec,hGraph,phErrorNode,logBuffer,bufferSize);
}

CUresult hacked_cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags) {
	LOG_DEBUG("cuGraphInstantiateWithFlags");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphInstantiateWithFlags,phGraphExec,hGraph,flags);
}

CUresult hacked_cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
	LOG_DEBUG("cuGraphUpload");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphUpload,hGraphExec,hStream);
}

CUresult hacked_cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
	LOG_DEBUG("cuGraphLaunch");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphLaunch,hGraphExec,hStream);
}

CUresult hacked_cuGraphExecDestroy(CUgraphExec hGraphExec) {
	LOG_DEBUG("cuGraphExecDestroy");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphExecDestroy,hGraphExec);
}

CUresult hacked_cuGraphDestroy(CUgraph hGraph) {
	LOG_DEBUG("cuGraphDestroy");
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphDestroy,hGraph);
}
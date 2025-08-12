#ifndef __HOOK_H__
#define __HOOK_H__

typedef struct {
  void *fn_ptr;   // function pointer
  char *name;     //function name
} entry_t;

#define FIND_ENTRY(table, sym) ({ (table)[OVERRIDE_##sym].fn_ptr; })


#define CUDA_FUNCTIONS(X) \
    /* cuInit Part */ \
    X(cuInit)   \
    /* cuDeivce Part */ \
    X(cuDeviceGetAttribute)   \
    X(cuDeviceGet)   \
    X(cuDeviceGetCount)   \
    X(cuDeviceGetName)   \
    X(cuDeviceCanAccessPeer)   \
    X(cuDeviceGetP2PAttribute)   \
    X(cuDeviceGetByPCIBusId)   \
    X(cuDeviceGetPCIBusId)   \
    X(cuDeviceGetUuid)   \
    X(cuDeviceGetDefaultMemPool)   \
    X(cuDeviceGetLuid)   \
    X(cuDeviceGetMemPool)   \
    X(cuDeviceTotalMem_v2)   \
    X(cuDriverGetVersion)   \
    X(cuDeviceGetTexture1DLinearMaxWidth)   \
    X(cuDeviceSetMemPool)   \
    X(cuFlushGPUDirectRDMAWrites)   \
    \
    /* cuContext Part */ \
    X(cuDevicePrimaryCtxGetState)   \
    X(cuDevicePrimaryCtxRetain)   \
    X(cuDevicePrimaryCtxSetFlags_v2)   \
    X(cuDevicePrimaryCtxRelease_v2)   \
    X(cuCtxGetDevice)   \
    X(cuCtxCreate_v2)   \
    X(cuCtxCreate_v3)   \
    X(cuCtxDestroy_v2)   \
    X(cuCtxGetApiVersion)   \
    X(cuCtxGetCacheConfig)   \
    X(cuCtxGetCurrent)   \
    X(cuCtxGetFlags)   \
    X(cuCtxGetLimit)   \
    X(cuCtxGetSharedMemConfig)   \
    X(cuCtxGetStreamPriorityRange)   \
    X(cuCtxPopCurrent_v2)   \
    X(cuCtxPushCurrent_v2)   \
    X(cuCtxSetCacheConfig)   \
    X(cuCtxSetCurrent)   \
    X(cuCtxSetLimit)   \
    X(cuCtxSetSharedMemConfig)   \
    X(cuCtxSynchronize)   \
    /*X(cuCtxEnablePeerAccess)*/   \
    X(cuGetExportTable)   \
    \
    /* cuStream Part */ \
    X(cuStreamCreate)   \
    X(cuStreamDestroy_v2)   \
    X(cuStreamSynchronize)   \
    \
    /* cuMemory Part */     \
    X(cuArray3DCreate_v2)   \
    X(cuArrayCreate_v2)   \
    X(cuArrayDestroy)   \
    X(cuMemAlloc_v2)   \
    X(cuMemAllocHost_v2)   \
    X(cuMemAllocManaged)   \
    X(cuMemAllocPitch_v2)   \
    X(cuMemFree_v2)   \
    X(cuMemFreeHost)   \
    X(cuMemHostAlloc)   \
    X(cuMemHostRegister_v2)   \
    X(cuMemHostUnregister)   \
    X(cuMemcpyDtoH_v2)   \
    X(cuMemcpyHtoD_v2)   \
    X(cuMipmappedArrayCreate)   \
    X(cuMipmappedArrayDestroy)   \
    /* X(cuMemGetInfo_v2) cuMemGetInfo_v2 control by HOOK_MEMINFO_ENABLE  */   \
    X(cuMemcpy)   \
    X(cuPointerGetAttribute)   \
    X(cuPointerGetAttributes)   \
    X(cuPointerSetAttribute)   \
    X(cuIpcCloseMemHandle)   \
    X(cuIpcGetMemHandle)   \
    X(cuIpcOpenMemHandle_v2)   \
    X(cuMemGetAddressRange_v2)   \
    X(cuMemcpyAsync)   \
    X(cuMemcpyAtoD_v2)   \
    X(cuMemcpyDtoA_v2)   \
    X(cuMemcpyDtoD_v2)   \
    X(cuMemcpyDtoDAsync_v2)   \
    X(cuMemcpyDtoHAsync_v2)   \
    X(cuMemcpyHtoDAsync_v2)   \
    X(cuMemcpyPeer)   \
    X(cuMemcpyPeerAsync)   \
    X(cuMemsetD16_v2)   \
    X(cuMemsetD16Async)   \
    X(cuMemsetD2D16_v2)   \
    X(cuMemsetD2D16Async)   \
    X(cuMemsetD2D32_v2)   \
    X(cuMemsetD2D32Async)   \
    X(cuMemsetD2D8_v2)   \
    X(cuMemsetD2D8Async)   \
    X(cuMemsetD32_v2)   \
    X(cuMemsetD32Async)   \
    X(cuMemsetD8_v2)   \
    X(cuMemsetD8Async)   \
    X(cuMemAdvise)   \
    X(cuFuncSetCacheConfig)   \
    X(cuFuncSetSharedMemConfig)   \
    X(cuFuncGetAttribute)   \
    X(cuFuncSetAttribute)   \
    X(cuLaunchKernel)   \
    X(cuLaunchCooperativeKernel)   \
    \
    /* cuEvent Part */ \
    X(cuEventCreate)   \
    X(cuEventDestroy_v2)   \
    X(cuModuleLoad)   \
    X(cuModuleLoadData)   \
    X(cuModuleLoadDataEx)   \
    X(cuModuleLoadFatBinary)   \
    X(cuModuleGetFunction)   \
    X(cuModuleUnload)   \
    X(cuModuleGetGlobal_v2)   \
    X(cuModuleGetTexRef)   \
    X(cuModuleGetSurfRef)   \
    X(cuLinkAddData_v2)   \
    X(cuLinkCreate_v2)   \
    X(cuLinkAddFile_v2)   \
    X(cuLinkComplete)   \
    X(cuLinkDestroy)   \
    \
    /* Virtual Memory Part */\
    X(cuMemAddressReserve)   \
    X(cuMemCreate)   \
    X(cuMemMap)   \
    X(cuMemAllocAsync)   \
    X(cuMemFreeAsync)   \
    \
    /* cuda11.7 new api memory part */\
    X(cuMemHostGetDevicePointer_v2)   \
    X(cuMemHostGetFlags)   \
    X(cuMemPoolTrimTo)   \
    X(cuMemPoolSetAttribute)   \
    X(cuMemPoolGetAttribute)   \
    X(cuMemPoolSetAccess)   \
    X(cuMemPoolGetAccess)   \
    X(cuMemPoolCreate)   \
    X(cuMemPoolDestroy)   \
    X(cuMemAllocFromPoolAsync)   \
    X(cuMemPoolExportToShareableHandle)   \
    X(cuMemPoolImportFromShareableHandle)   \
    X(cuMemPoolExportPointer)   \
    X(cuMemPoolImportPointer)   \
    X(cuMemcpy2DUnaligned_v2)   \
    X(cuMemcpy2DAsync_v2)   \
    X(cuMemcpy3D_v2)   \
    X(cuMemcpy3DAsync_v2)   \
    X(cuMemcpy3DPeer)   \
    X(cuMemcpy3DPeerAsync)   \
    X(cuMemPrefetchAsync)   \
    X(cuMemRangeGetAttribute)   \
    X(cuMemRangeGetAttributes)   \
    \
    /* cuda 11.7 external resource management */\
    X(cuImportExternalMemory)   \
    X(cuExternalMemoryGetMappedBuffer)   \
    X(cuExternalMemoryGetMappedMipmappedArray)   \
    X(cuDestroyExternalMemory)   \
    X(cuImportExternalSemaphore)   \
    X(cuSignalExternalSemaphoresAsync)   \
    X(cuWaitExternalSemaphoresAsync)   \
    X(cuDestroyExternalSemaphore)   \
    \
    /* cuda graph part */\
    X(cuGraphCreate)   \
    X(cuGraphAddKernelNode_v2)   \
    X(cuGraphKernelNodeGetParams_v2)   \
    X(cuGraphKernelNodeSetParams_v2)   \
    X(cuGraphAddMemcpyNode)   \
    X(cuGraphMemcpyNodeGetParams)   \
    X(cuGraphMemcpyNodeSetParams)   \
    X(cuGraphAddMemsetNode)   \
    X(cuGraphMemsetNodeGetParams)   \
    X(cuGraphMemsetNodeSetParams)   \
    X(cuGraphAddHostNode)   \
    X(cuGraphHostNodeGetParams)   \
    X(cuGraphHostNodeSetParams)   \
    X(cuGraphAddChildGraphNode)   \
    X(cuGraphChildGraphNodeGetGraph)   \
    X(cuGraphAddEmptyNode)   \
    X(cuGraphAddEventRecordNode)   \
    X(cuGraphEventRecordNodeGetEvent)   \
    X(cuGraphEventRecordNodeSetEvent)   \
    X(cuGraphAddEventWaitNode)   \
    X(cuGraphEventWaitNodeGetEvent)   \
    X(cuGraphEventWaitNodeSetEvent)   \
    X(cuGraphAddExternalSemaphoresSignalNode)   \
    X(cuGraphExternalSemaphoresSignalNodeGetParams)   \
    X(cuGraphExternalSemaphoresSignalNodeSetParams)   \
    X(cuGraphAddExternalSemaphoresWaitNode)   \
    X(cuGraphExternalSemaphoresWaitNodeGetParams)   \
    X(cuGraphExternalSemaphoresWaitNodeSetParams)   \
    X(cuGraphExecExternalSemaphoresSignalNodeSetParams)   \
    X(cuGraphExecExternalSemaphoresWaitNodeSetParams)   \
    X(cuGraphClone)   \
    X(cuGraphNodeFindInClone)   \
    X(cuGraphNodeGetType)   \
    X(cuGraphGetNodes)   \
    X(cuGraphGetRootNodes)   \
    X(cuGraphGetEdges)   \
    X(cuGraphNodeGetDependencies)   \
    X(cuGraphNodeGetDependentNodes)   \
    X(cuGraphAddDependencies)   \
    X(cuGraphRemoveDependencies)   \
    X(cuGraphDestroyNode)   \
    X(cuGraphInstantiate)   \
    X(cuGraphInstantiateWithFlags)   \
    X(cuGraphUpload)   \
    X(cuGraphLaunch)   \
    X(cuGraphExecDestroy)   \
    X(cuGraphDestroy)   \
    \
    X(cuGetProcAddress)   \
    X(cuGetProcAddress_v2)   \


#define NVML_FUNCTIONS(X) \
    /* init */ \
    X(nvmlInit) \
    X(nvmlInit_v2) \
    X(nvmlInitWithFlags) \
    X(nvmlShutdown) \
    X(nvmlErrorString) \
    \
    /* device part */ \
    X(nvmlDeviceGetHandleByIndex) \
    X(nvmlDeviceGetComputeRunningProcesses) \
    X(nvmlDeviceGetPciInfo) \
    X(nvmlDeviceGetProcessUtilization) \
    X(nvmlDeviceGetCount) \
    X(nvmlDeviceClearAccountingPids) \
    X(nvmlDeviceClearCpuAffinity) \
    X(nvmlDeviceClearEccErrorCounts) \
    X(nvmlDeviceDiscoverGpus) \
    X(nvmlDeviceFreezeNvLinkUtilizationCounter) \
    X(nvmlDeviceGetAccountingBufferSize) \
    X(nvmlDeviceGetAccountingMode) \
    X(nvmlDeviceGetAccountingPids) \
    X(nvmlDeviceGetAccountingStats) \
    X(nvmlDeviceGetActiveVgpus) \
    X(nvmlDeviceGetAPIRestriction) \
    X(nvmlDeviceGetApplicationsClock) \
    X(nvmlDeviceGetAutoBoostedClocksEnabled) \
    X(nvmlDeviceGetBAR1MemoryInfo) \
    X(nvmlDeviceGetBoardId) \
    X(nvmlDeviceGetBoardPartNumber) \
    X(nvmlDeviceGetBrand) \
    X(nvmlDeviceGetBridgeChipInfo) \
    X(nvmlDeviceGetClock) \
    X(nvmlDeviceGetClockInfo) \
    X(nvmlDeviceGetComputeMode) \
    X(nvmlDeviceGetCount_v2) \
    X(nvmlDeviceGetCpuAffinity) \
    X(nvmlDeviceGetCreatableVgpus) \
    X(nvmlDeviceGetCudaComputeCapability) \
    X(nvmlDeviceGetCurrentClocksThrottleReasons) \
    X(nvmlDeviceGetCurrPcieLinkGeneration) \
    X(nvmlDeviceGetCurrPcieLinkWidth) \
    X(nvmlDeviceGetDecoderUtilization) \
    X(nvmlDeviceGetDefaultApplicationsClock) \
    X(nvmlDeviceGetDetailedEccErrors) \
    X(nvmlDeviceGetDisplayActive) \
    X(nvmlDeviceGetDisplayMode) \
    X(nvmlDeviceGetDriverModel) \
    X(nvmlDeviceGetEccMode) \
    X(nvmlDeviceGetEncoderCapacity) \
    X(nvmlDeviceGetEncoderSessions) \
    X(nvmlDeviceGetEncoderStats) \
    X(nvmlDeviceGetEncoderUtilization) \
    X(nvmlDeviceGetEnforcedPowerLimit) \
    X(nvmlDeviceGetFanSpeed) \
    X(nvmlDeviceGetFanSpeed_v2) \
    X(nvmlDeviceGetFieldValues) \
    X(nvmlDeviceGetGpuOperationMode) \
    X(nvmlDeviceGetGraphicsRunningProcesses) \
    X(nvmlDeviceGetGridLicensableFeatures) \
    X(nvmlDeviceGetHandleByIndex_v2) \
    X(nvmlDeviceGetHandleByPciBusId) \
    X(nvmlDeviceGetHandleByPciBusId_v2) \
    X(nvmlDeviceGetHandleBySerial) \
    X(nvmlDeviceGetHandleByUUID) \
    X(nvmlDeviceGetIndex) \
    X(nvmlDeviceGetInforomConfigurationChecksum) \
    X(nvmlDeviceGetInforomImageVersion) \
    X(nvmlDeviceGetInforomVersion) \
    X(nvmlDeviceGetMaxClockInfo) \
    X(nvmlDeviceGetMaxCustomerBoostClock) \
    X(nvmlDeviceGetMaxPcieLinkGeneration) \
    X(nvmlDeviceGetMaxPcieLinkWidth) \
    X(nvmlDeviceGetMemoryErrorCounter) \
    X(nvmlDeviceGetMemoryInfo) \
    X(nvmlDeviceGetMemoryInfo_v2) \
    X(nvmlDeviceGetMinorNumber) \
    X(nvmlDeviceGetMPSComputeRunningProcesses) \
    X(nvmlDeviceGetMultiGpuBoard) \
    X(nvmlDeviceGetName) \
    X(nvmlDeviceGetNvLinkCapability) \
    X(nvmlDeviceGetNvLinkErrorCounter) \
    X(nvmlDeviceGetNvLinkRemotePciInfo) \
    X(nvmlDeviceGetNvLinkRemotePciInfo_v2) \
    X(nvmlDeviceGetNvLinkState) \
    X(nvmlDeviceGetNvLinkUtilizationControl) \
    X(nvmlDeviceGetNvLinkUtilizationCounter) \
    X(nvmlDeviceGetNvLinkVersion) \
    X(nvmlDeviceGetP2PStatus) \
    X(nvmlDeviceGetPcieReplayCounter) \
    X(nvmlDeviceGetPcieThroughput) \
    X(nvmlDeviceGetPciInfo_v2) \
    X(nvmlDeviceGetPciInfo_v3) \
    X(nvmlDeviceGetPerformanceState) \
    X(nvmlDeviceGetPersistenceMode) \
    X(nvmlDeviceGetPowerManagementDefaultLimit) \
    X(nvmlDeviceGetPowerManagementLimit) \
    X(nvmlDeviceGetPowerManagementLimitConstraints) \
    X(nvmlDeviceGetPowerManagementMode) \
    X(nvmlDeviceGetPowerState) \
    X(nvmlDeviceGetPowerUsage) \
    X(nvmlDeviceGetRetiredPages) \
    X(nvmlDeviceGetRetiredPagesPendingStatus) \
    X(nvmlDeviceGetSamples) \
    X(nvmlDeviceGetSerial) \
    X(nvmlDeviceGetSupportedClocksThrottleReasons) \
    X(nvmlDeviceGetSupportedEventTypes) \
    X(nvmlDeviceGetSupportedGraphicsClocks) \
    X(nvmlDeviceGetSupportedMemoryClocks) \
    X(nvmlDeviceGetSupportedVgpus) \
    X(nvmlDeviceGetTemperature) \
    X(nvmlDeviceGetTemperatureThreshold) \
    X(nvmlDeviceGetTopologyCommonAncestor) \
    X(nvmlDeviceGetTopologyNearestGpus) \
    X(nvmlDeviceGetTotalEccErrors) \
    X(nvmlDeviceGetTotalEnergyConsumption) \
    X(nvmlDeviceGetUtilizationRates) \
    X(nvmlDeviceGetUUID) \
    X(nvmlDeviceGetVbiosVersion) \
    X(nvmlDeviceGetVgpuMetadata) \
    X(nvmlDeviceGetVgpuProcessUtilization) \
    X(nvmlDeviceGetVgpuUtilization) \
    X(nvmlDeviceGetViolationStatus) \
    X(nvmlDeviceGetVirtualizationMode) \
    X(nvmlDeviceModifyDrainState) \
    X(nvmlDeviceOnSameBoard) \
    X(nvmlDeviceQueryDrainState) \
    X(nvmlDeviceRegisterEvents) \
    X(nvmlDeviceRemoveGpu) \
    X(nvmlDeviceRemoveGpu_v2) \
    X(nvmlDeviceResetApplicationsClocks) \
    X(nvmlDeviceResetNvLinkErrorCounters) \
    X(nvmlDeviceResetNvLinkUtilizationCounter) \
    X(nvmlDeviceSetAccountingMode) \
    X(nvmlDeviceSetAPIRestriction) \
    X(nvmlDeviceSetApplicationsClocks) \
    X(nvmlDeviceSetAutoBoostedClocksEnabled) \
    X(nvmlDeviceSetComputeMode) \
    X(nvmlDeviceSetCpuAffinity) \
    X(nvmlDeviceSetDefaultAutoBoostedClocksEnabled) \
    X(nvmlDeviceSetDriverModel) \
    X(nvmlDeviceSetEccMode) \
    X(nvmlDeviceSetGpuOperationMode) \
    X(nvmlDeviceSetNvLinkUtilizationControl) \
    X(nvmlDeviceSetPersistenceMode) \
    X(nvmlDeviceSetPowerManagementLimit) \
    X(nvmlDeviceSetVirtualizationMode) \
    X(nvmlDeviceValidateInforom) \
    X(nvmlDeviceGetComputeRunningProcesses_v2) \
    X(nvmlDeviceGetGraphicsRunningProcesses_v2) \
    X(nvmlDeviceSetTemperatureThreshold) \
    X(nvmlDeviceGetFBCSessions) \
    X(nvmlDeviceGetFBCStats) \
    X(nvmlDeviceGetGridLicensableFeatures_v2) \
    X(nvmlDeviceGetRetiredPages_v2) \
    X(nvmlDeviceResetGpuLockedClocks) \
    X(nvmlDeviceSetGpuLockedClocks) \
    X(nvmlDeviceCreateGpuInstance) \
    X(nvmlDeviceGetArchitecture) \
    X(nvmlDeviceGetAttributes) \
    X(nvmlDeviceGetAttributes_v2) \
    X(nvmlDeviceGetComputeInstanceId) \
    X(nvmlDeviceGetCpuAffinityWithinScope) \
    X(nvmlDeviceGetDeviceHandleFromMigDeviceHandle) \
    X(nvmlDeviceGetGpuInstanceById) \
    X(nvmlDeviceGetGpuInstanceId) \
    X(nvmlDeviceGetGpuInstancePossiblePlacements) \
    X(nvmlDeviceGetGpuInstanceProfileInfo) \
    X(nvmlDeviceGetGpuInstanceRemainingCapacity) \
    X(nvmlDeviceGetGpuInstances) \
    X(nvmlDeviceGetMaxMigDeviceCount) \
    X(nvmlDeviceGetMemoryAffinity) \
    X(nvmlDeviceGetMigDeviceHandleByIndex) \
    X(nvmlDeviceGetMigMode) \
    X(nvmlDeviceGetRemappedRows) \
    X(nvmlDeviceGetRowRemapperHistogram) \
    X(nvmlDeviceIsMigDeviceHandle) \
    X(nvmlDeviceSetMigMode) \
    X(nvmlDeviceGetGridLicensableFeatures_v3) \
    X(nvmlDeviceGetHostVgpuMode) \
    X(nvmlDeviceGetPgpuMetadataString) \
    \
    /* unit */ \
    X(nvmlUnitGetCount) \
    X(nvmlUnitGetDevices) \
    X(nvmlUnitGetFanSpeedInfo) \
    X(nvmlUnitGetHandleByIndex) \
    X(nvmlUnitGetLedState) \
    X(nvmlUnitGetPsuInfo) \
    X(nvmlUnitGetTemperature) \
    X(nvmlUnitGetUnitInfo) \
    X(nvmlUnitSetLedState) \
    \
    /* system part */ \
    X(nvmlSystemGetCudaDriverVersion) \
    X(nvmlSystemGetCudaDriverVersion_v2) \
    X(nvmlSystemGetDriverVersion) \
    X(nvmlSystemGetHicVersion) \
    X(nvmlSystemGetNVMLVersion) \
    X(nvmlSystemGetProcessName) \
    X(nvmlSystemGetTopologyGpuSet) \
    \
    /* internal */ \
    X(nvmlInternalGetExportTable) \
    \
    /* compute instance */ \
    X(nvmlComputeInstanceDestroy) \
    X(nvmlComputeInstanceGetInfo) \
    \
    X(nvmlGpuInstanceCreateComputeInstance) \
    X(nvmlGpuInstanceDestroy) \
    X(nvmlGpuInstanceGetComputeInstanceById) \
    X(nvmlGpuInstanceGetComputeInstanceProfileInfo) \
    X(nvmlGpuInstanceGetComputeInstanceRemainingCapacity) \
    X(nvmlGpuInstanceGetComputeInstances) \
    X(nvmlGpuInstanceGetInfo) \
    X(nvmlComputeInstanceGetInfo_v2) \
    \
    /* vgpu part */ \
    X(nvmlGetVgpuVersion) \
    X(nvmlSetVgpuVersion) \
    X(nvmlVgpuInstanceGetEccMode) \
    X(nvmlVgpuInstanceClearAccountingPids) \
    X(nvmlVgpuInstanceGetMdevUUID) \
    X(nvmlVgpuInstanceGetAccountingMode) \
    X(nvmlVgpuInstanceGetAccountingPids) \
    X(nvmlVgpuInstanceGetAccountingStats) \
    X(nvmlVgpuInstanceGetFBCSessions) \
    X(nvmlVgpuInstanceGetFBCStats) \
    X(nvmlVgpuTypeGetMaxInstancesPerVm) \
    X(nvmlVgpuInstanceGetGpuInstanceId) \
    X(nvmlVgpuTypeGetGpuInstanceProfileId) \
    X(nvmlGetVgpuCompatibility) \
    X(nvmlVgpuInstanceGetEncoderCapacity) \
    X(nvmlVgpuInstanceGetEncoderSessions) \
    X(nvmlVgpuInstanceGetEncoderStats) \
    X(nvmlVgpuInstanceGetFbUsage) \
    X(nvmlVgpuInstanceGetFrameRateLimit) \
    X(nvmlVgpuInstanceGetLicenseStatus) \
    X(nvmlVgpuInstanceGetMetadata) \
    X(nvmlVgpuInstanceGetType) \
    X(nvmlVgpuInstanceGetUUID) \
    X(nvmlVgpuInstanceGetVmDriverVersion) \
    X(nvmlVgpuInstanceGetVmID) \
    X(nvmlVgpuInstanceSetEncoderCapacity) \
    X(nvmlVgpuTypeGetClass) \
    X(nvmlVgpuTypeGetDeviceID) \
    X(nvmlVgpuTypeGetFramebufferSize) \
    X(nvmlVgpuTypeGetFrameRateLimit) \
    X(nvmlVgpuTypeGetLicense) \
    X(nvmlVgpuTypeGetMaxInstances) \
    X(nvmlVgpuTypeGetName) \
    X(nvmlVgpuTypeGetNumDisplayHeads) \
    X(nvmlVgpuTypeGetResolution) \
    \
    /* event part */ \
    X(nvmlEventSetCreate) \
    X(nvmlEventSetFree) \
    X(nvmlEventSetWait) \
    X(nvmlEventSetWait_v2)

    
#endif // __HOOK_H__
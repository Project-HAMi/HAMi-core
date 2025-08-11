#ifndef __LIBNVML_HOOK_H__
#define __LIBNVML_HOOK_H__

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda.h>
#include <pthread.h>

#include "include/hook.h"
#include "include/nvml-subset.h"
#include "include/log_utils.h"
#include "include/nvml_prefix.h"

#define FILENAME_MAX 4096

typedef nvmlReturn_t (*driver_sym_t)();

#define NVML_OVERRIDE_CALL(table, sym, ...)                                    \
  ({                                                                           \
    LOG_DEBUG("Hijacking %s", #sym);                                           \
    driver_sym_t _entry = FIND_ENTRY(table, sym);                         \
    _entry(__VA_ARGS__);                                                       \
  })

#define NVML_OVERRIDE_CALL_NO_LOG(table, sym, ...)                             \
  ({                                                                           \
    driver_sym_t _entry = FIND_ENTRY(table, sym);                         \
    _entry(__VA_ARGS__);                                                       \
  })

/**
 * NVML management library enumerator entry
 */
typedef enum {
  /* init */
  OVERRIDE_ENUM(nvmlInit),
  OVERRIDE_ENUM(nvmlInit_v2),
  OVERRIDE_ENUM(nvmlInitWithFlags),
  OVERRIDE_ENUM(nvmlShutdown),
  OVERRIDE_ENUM(nvmlErrorString),

  /* device part */
  OVERRIDE_ENUM(nvmlDeviceGetHandleByIndex),
  OVERRIDE_ENUM(nvmlDeviceGetComputeRunningProcesses),
  OVERRIDE_ENUM(nvmlDeviceGetPciInfo),
  OVERRIDE_ENUM(nvmlDeviceGetProcessUtilization),
  OVERRIDE_ENUM(nvmlDeviceGetCount),
  OVERRIDE_ENUM(nvmlDeviceClearAccountingPids),
  OVERRIDE_ENUM(nvmlDeviceClearCpuAffinity),
  OVERRIDE_ENUM(nvmlDeviceClearEccErrorCounts),
  OVERRIDE_ENUM(nvmlDeviceDiscoverGpus),
  OVERRIDE_ENUM(nvmlDeviceFreezeNvLinkUtilizationCounter),
  OVERRIDE_ENUM(nvmlDeviceGetAccountingBufferSize),
  OVERRIDE_ENUM(nvmlDeviceGetAccountingMode),
  OVERRIDE_ENUM(nvmlDeviceGetAccountingPids),
  OVERRIDE_ENUM(nvmlDeviceGetAccountingStats),
  OVERRIDE_ENUM(nvmlDeviceGetActiveVgpus),
  OVERRIDE_ENUM(nvmlDeviceGetAPIRestriction),
  OVERRIDE_ENUM(nvmlDeviceGetApplicationsClock),
  OVERRIDE_ENUM(nvmlDeviceGetAutoBoostedClocksEnabled),
  OVERRIDE_ENUM(nvmlDeviceGetBAR1MemoryInfo),
  OVERRIDE_ENUM(nvmlDeviceGetBoardId),
  OVERRIDE_ENUM(nvmlDeviceGetBoardPartNumber),
  OVERRIDE_ENUM(nvmlDeviceGetBrand),
  OVERRIDE_ENUM(nvmlDeviceGetBridgeChipInfo),
  OVERRIDE_ENUM(nvmlDeviceGetClock),
  OVERRIDE_ENUM(nvmlDeviceGetClockInfo),
  OVERRIDE_ENUM(nvmlDeviceGetComputeMode),
  OVERRIDE_ENUM(nvmlDeviceGetCount_v2),
  OVERRIDE_ENUM(nvmlDeviceGetCpuAffinity),
  OVERRIDE_ENUM(nvmlDeviceGetCreatableVgpus),
  OVERRIDE_ENUM(nvmlDeviceGetCudaComputeCapability),
  OVERRIDE_ENUM(nvmlDeviceGetCurrentClocksThrottleReasons),
  OVERRIDE_ENUM(nvmlDeviceGetCurrPcieLinkGeneration),
  OVERRIDE_ENUM(nvmlDeviceGetCurrPcieLinkWidth),
  OVERRIDE_ENUM(nvmlDeviceGetDecoderUtilization),
  OVERRIDE_ENUM(nvmlDeviceGetDefaultApplicationsClock),
  OVERRIDE_ENUM(nvmlDeviceGetDetailedEccErrors),
  OVERRIDE_ENUM(nvmlDeviceGetDisplayActive),
  OVERRIDE_ENUM(nvmlDeviceGetDisplayMode),
  OVERRIDE_ENUM(nvmlDeviceGetDriverModel),
  OVERRIDE_ENUM(nvmlDeviceGetEccMode),
  OVERRIDE_ENUM(nvmlDeviceGetEncoderCapacity),
  OVERRIDE_ENUM(nvmlDeviceGetEncoderSessions),
  OVERRIDE_ENUM(nvmlDeviceGetEncoderStats),
  OVERRIDE_ENUM(nvmlDeviceGetEncoderUtilization),
  OVERRIDE_ENUM(nvmlDeviceGetEnforcedPowerLimit),
  OVERRIDE_ENUM(nvmlDeviceGetFanSpeed),
  OVERRIDE_ENUM(nvmlDeviceGetFanSpeed_v2),
  OVERRIDE_ENUM(nvmlDeviceGetFieldValues),
  OVERRIDE_ENUM(nvmlDeviceGetGpuOperationMode),
  OVERRIDE_ENUM(nvmlDeviceGetGraphicsRunningProcesses),
  OVERRIDE_ENUM(nvmlDeviceGetGridLicensableFeatures),
  OVERRIDE_ENUM(nvmlDeviceGetHandleByIndex_v2),
  OVERRIDE_ENUM(nvmlDeviceGetHandleByPciBusId),
  OVERRIDE_ENUM(nvmlDeviceGetHandleByPciBusId_v2),
  OVERRIDE_ENUM(nvmlDeviceGetHandleBySerial),
  OVERRIDE_ENUM(nvmlDeviceGetHandleByUUID),
  OVERRIDE_ENUM(nvmlDeviceGetIndex),
  OVERRIDE_ENUM(nvmlDeviceGetInforomConfigurationChecksum),
  OVERRIDE_ENUM(nvmlDeviceGetInforomImageVersion),
  OVERRIDE_ENUM(nvmlDeviceGetInforomVersion),
  OVERRIDE_ENUM(nvmlDeviceGetMaxClockInfo),
  OVERRIDE_ENUM(nvmlDeviceGetMaxCustomerBoostClock),
  OVERRIDE_ENUM(nvmlDeviceGetMaxPcieLinkGeneration),
  OVERRIDE_ENUM(nvmlDeviceGetMaxPcieLinkWidth),
  OVERRIDE_ENUM(nvmlDeviceGetMemoryErrorCounter),
  OVERRIDE_ENUM(nvmlDeviceGetMemoryInfo),
  OVERRIDE_ENUM(nvmlDeviceGetMemoryInfo_v2),
  OVERRIDE_ENUM(nvmlDeviceGetMinorNumber),
  OVERRIDE_ENUM(nvmlDeviceGetMPSComputeRunningProcesses),
  OVERRIDE_ENUM(nvmlDeviceGetMultiGpuBoard),
  OVERRIDE_ENUM(nvmlDeviceGetName),
  OVERRIDE_ENUM(nvmlDeviceGetNvLinkCapability),
  OVERRIDE_ENUM(nvmlDeviceGetNvLinkErrorCounter),
  OVERRIDE_ENUM(nvmlDeviceGetNvLinkRemotePciInfo),
  OVERRIDE_ENUM(nvmlDeviceGetNvLinkRemotePciInfo_v2),
  OVERRIDE_ENUM(nvmlDeviceGetNvLinkState),
  OVERRIDE_ENUM(nvmlDeviceGetNvLinkUtilizationControl),
  OVERRIDE_ENUM(nvmlDeviceGetNvLinkUtilizationCounter),
  OVERRIDE_ENUM(nvmlDeviceGetNvLinkVersion),
  OVERRIDE_ENUM(nvmlDeviceGetP2PStatus),
  OVERRIDE_ENUM(nvmlDeviceGetPcieReplayCounter),
  OVERRIDE_ENUM(nvmlDeviceGetPcieThroughput),
  OVERRIDE_ENUM(nvmlDeviceGetPciInfo_v2),
  OVERRIDE_ENUM(nvmlDeviceGetPciInfo_v3),
  OVERRIDE_ENUM(nvmlDeviceGetPerformanceState),
  OVERRIDE_ENUM(nvmlDeviceGetPersistenceMode),
  OVERRIDE_ENUM(nvmlDeviceGetPowerManagementDefaultLimit),
  OVERRIDE_ENUM(nvmlDeviceGetPowerManagementLimit),
  OVERRIDE_ENUM(nvmlDeviceGetPowerManagementLimitConstraints),
  OVERRIDE_ENUM(nvmlDeviceGetPowerManagementMode),
  OVERRIDE_ENUM(nvmlDeviceGetPowerState),
  OVERRIDE_ENUM(nvmlDeviceGetPowerUsage),
  OVERRIDE_ENUM(nvmlDeviceGetRetiredPages),
  OVERRIDE_ENUM(nvmlDeviceGetRetiredPagesPendingStatus),
  OVERRIDE_ENUM(nvmlDeviceGetSamples),
  OVERRIDE_ENUM(nvmlDeviceGetSerial),
  OVERRIDE_ENUM(nvmlDeviceGetSupportedClocksThrottleReasons),
  OVERRIDE_ENUM(nvmlDeviceGetSupportedEventTypes),
  OVERRIDE_ENUM(nvmlDeviceGetSupportedGraphicsClocks),
  OVERRIDE_ENUM(nvmlDeviceGetSupportedMemoryClocks),
  OVERRIDE_ENUM(nvmlDeviceGetSupportedVgpus),
  OVERRIDE_ENUM(nvmlDeviceGetTemperature),
  OVERRIDE_ENUM(nvmlDeviceGetTemperatureThreshold),
  OVERRIDE_ENUM(nvmlDeviceGetTopologyCommonAncestor),
  OVERRIDE_ENUM(nvmlDeviceGetTopologyNearestGpus),
  OVERRIDE_ENUM(nvmlDeviceGetTotalEccErrors),
  OVERRIDE_ENUM(nvmlDeviceGetTotalEnergyConsumption),
  OVERRIDE_ENUM(nvmlDeviceGetUtilizationRates),
  OVERRIDE_ENUM(nvmlDeviceGetUUID),
  OVERRIDE_ENUM(nvmlDeviceGetVbiosVersion),
  OVERRIDE_ENUM(nvmlDeviceGetVgpuMetadata),
  OVERRIDE_ENUM(nvmlDeviceGetVgpuProcessUtilization),
  OVERRIDE_ENUM(nvmlDeviceGetVgpuUtilization),
  OVERRIDE_ENUM(nvmlDeviceGetViolationStatus),
  OVERRIDE_ENUM(nvmlDeviceGetVirtualizationMode),
  OVERRIDE_ENUM(nvmlDeviceModifyDrainState),
  OVERRIDE_ENUM(nvmlDeviceOnSameBoard),
  OVERRIDE_ENUM(nvmlDeviceQueryDrainState),
  OVERRIDE_ENUM(nvmlDeviceRegisterEvents),
  OVERRIDE_ENUM(nvmlDeviceRemoveGpu),
  OVERRIDE_ENUM(nvmlDeviceRemoveGpu_v2),
  OVERRIDE_ENUM(nvmlDeviceResetApplicationsClocks),
  OVERRIDE_ENUM(nvmlDeviceResetNvLinkErrorCounters),
  OVERRIDE_ENUM(nvmlDeviceResetNvLinkUtilizationCounter),
  OVERRIDE_ENUM(nvmlDeviceSetAccountingMode),
  OVERRIDE_ENUM(nvmlDeviceSetAPIRestriction),
  OVERRIDE_ENUM(nvmlDeviceSetApplicationsClocks),
  OVERRIDE_ENUM(nvmlDeviceSetAutoBoostedClocksEnabled),
  OVERRIDE_ENUM(nvmlDeviceSetComputeMode),
  OVERRIDE_ENUM(nvmlDeviceSetCpuAffinity),
  OVERRIDE_ENUM(nvmlDeviceSetDefaultAutoBoostedClocksEnabled),
  OVERRIDE_ENUM(nvmlDeviceSetDriverModel),
  OVERRIDE_ENUM(nvmlDeviceSetEccMode),
  OVERRIDE_ENUM(nvmlDeviceSetGpuOperationMode),
  OVERRIDE_ENUM(nvmlDeviceSetNvLinkUtilizationControl),
  OVERRIDE_ENUM(nvmlDeviceSetPersistenceMode),
  OVERRIDE_ENUM(nvmlDeviceSetPowerManagementLimit),
  OVERRIDE_ENUM(nvmlDeviceSetVirtualizationMode),
  OVERRIDE_ENUM(nvmlDeviceValidateInforom),
  OVERRIDE_ENUM(nvmlDeviceGetComputeRunningProcesses_v2),
  OVERRIDE_ENUM(nvmlDeviceGetGraphicsRunningProcesses_v2),
  OVERRIDE_ENUM(nvmlDeviceSetTemperatureThreshold),
  OVERRIDE_ENUM(nvmlDeviceGetFBCSessions),
  OVERRIDE_ENUM(nvmlDeviceGetFBCStats),
  OVERRIDE_ENUM(nvmlDeviceGetGridLicensableFeatures_v2),
  OVERRIDE_ENUM(nvmlDeviceGetRetiredPages_v2),
  OVERRIDE_ENUM(nvmlDeviceResetGpuLockedClocks),
  OVERRIDE_ENUM(nvmlDeviceSetGpuLockedClocks),
  OVERRIDE_ENUM(nvmlDeviceCreateGpuInstance),
  OVERRIDE_ENUM(nvmlDeviceGetArchitecture),
  OVERRIDE_ENUM(nvmlDeviceGetAttributes),
  OVERRIDE_ENUM(nvmlDeviceGetAttributes_v2),
  OVERRIDE_ENUM(nvmlDeviceGetComputeInstanceId),
  OVERRIDE_ENUM(nvmlDeviceGetCpuAffinityWithinScope),
  OVERRIDE_ENUM(nvmlDeviceGetDeviceHandleFromMigDeviceHandle),
  OVERRIDE_ENUM(nvmlDeviceGetGpuInstanceById),
  OVERRIDE_ENUM(nvmlDeviceGetGpuInstanceId),
  OVERRIDE_ENUM(nvmlDeviceGetGpuInstancePossiblePlacements),
  OVERRIDE_ENUM(nvmlDeviceGetGpuInstanceProfileInfo),
  OVERRIDE_ENUM(nvmlDeviceGetGpuInstanceRemainingCapacity),
  OVERRIDE_ENUM(nvmlDeviceGetGpuInstances),
  OVERRIDE_ENUM(nvmlDeviceGetMaxMigDeviceCount),
  OVERRIDE_ENUM(nvmlDeviceGetMemoryAffinity),
  OVERRIDE_ENUM(nvmlDeviceGetMigDeviceHandleByIndex),
  OVERRIDE_ENUM(nvmlDeviceGetMigMode),
  OVERRIDE_ENUM(nvmlDeviceGetRemappedRows),
  OVERRIDE_ENUM(nvmlDeviceGetRowRemapperHistogram),
  OVERRIDE_ENUM(nvmlDeviceIsMigDeviceHandle),
  OVERRIDE_ENUM(nvmlDeviceSetMigMode),
  OVERRIDE_ENUM(nvmlDeviceGetGridLicensableFeatures_v3),
  OVERRIDE_ENUM(nvmlDeviceGetHostVgpuMode),
  OVERRIDE_ENUM(nvmlDeviceGetPgpuMetadataString),

  /* unit */
  OVERRIDE_ENUM(nvmlUnitGetCount),
  OVERRIDE_ENUM(nvmlUnitGetDevices),
  OVERRIDE_ENUM(nvmlUnitGetFanSpeedInfo),
  OVERRIDE_ENUM(nvmlUnitGetHandleByIndex),
  OVERRIDE_ENUM(nvmlUnitGetLedState),
  OVERRIDE_ENUM(nvmlUnitGetPsuInfo),
  OVERRIDE_ENUM(nvmlUnitGetTemperature),
  OVERRIDE_ENUM(nvmlUnitGetUnitInfo),
  OVERRIDE_ENUM(nvmlUnitSetLedState),


  /* system part */
  OVERRIDE_ENUM(nvmlSystemGetCudaDriverVersion),
  OVERRIDE_ENUM(nvmlSystemGetCudaDriverVersion_v2),
  OVERRIDE_ENUM(nvmlSystemGetDriverVersion),
  OVERRIDE_ENUM(nvmlSystemGetHicVersion),
  OVERRIDE_ENUM(nvmlSystemGetNVMLVersion),
  OVERRIDE_ENUM(nvmlSystemGetProcessName),
  OVERRIDE_ENUM(nvmlSystemGetTopologyGpuSet),

  /* internal */
  OVERRIDE_ENUM(nvmlInternalGetExportTable),
  
  /* compute instance */
  OVERRIDE_ENUM(nvmlComputeInstanceDestroy),
  OVERRIDE_ENUM(nvmlComputeInstanceGetInfo),


  OVERRIDE_ENUM(nvmlGpuInstanceCreateComputeInstance),
  OVERRIDE_ENUM(nvmlGpuInstanceDestroy),
  OVERRIDE_ENUM(nvmlGpuInstanceGetComputeInstanceById),
  OVERRIDE_ENUM(nvmlGpuInstanceGetComputeInstanceProfileInfo),
  OVERRIDE_ENUM(nvmlGpuInstanceGetComputeInstanceRemainingCapacity),
  OVERRIDE_ENUM(nvmlGpuInstanceGetComputeInstances),
  OVERRIDE_ENUM(nvmlGpuInstanceGetInfo),
  OVERRIDE_ENUM(nvmlComputeInstanceGetInfo_v2),

  //OVERRIDE_ENUM(nvmlRetry_NvRmControl),

  /* vgpu part */
  OVERRIDE_ENUM(nvmlGetVgpuVersion),
  OVERRIDE_ENUM(nvmlSetVgpuVersion),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetEccMode),
  OVERRIDE_ENUM(nvmlVgpuInstanceClearAccountingPids),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetMdevUUID),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetAccountingMode),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetAccountingPids),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetAccountingStats),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetFBCSessions),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetFBCStats),
  OVERRIDE_ENUM(nvmlVgpuTypeGetMaxInstancesPerVm),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetGpuInstanceId),
  OVERRIDE_ENUM(nvmlVgpuTypeGetGpuInstanceProfileId),
  OVERRIDE_ENUM(nvmlGetVgpuCompatibility),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetEncoderCapacity),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetEncoderSessions),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetEncoderStats),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetFbUsage),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetFrameRateLimit),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetLicenseStatus),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetMetadata),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetType),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetUUID),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetVmDriverVersion),
  OVERRIDE_ENUM(nvmlVgpuInstanceGetVmID),
  OVERRIDE_ENUM(nvmlVgpuInstanceSetEncoderCapacity),
  OVERRIDE_ENUM(nvmlVgpuTypeGetClass),
  OVERRIDE_ENUM(nvmlVgpuTypeGetDeviceID),
  OVERRIDE_ENUM(nvmlVgpuTypeGetFramebufferSize),
  OVERRIDE_ENUM(nvmlVgpuTypeGetFrameRateLimit),
  OVERRIDE_ENUM(nvmlVgpuTypeGetLicense),
  OVERRIDE_ENUM(nvmlVgpuTypeGetMaxInstances),
  OVERRIDE_ENUM(nvmlVgpuTypeGetName),
  OVERRIDE_ENUM(nvmlVgpuTypeGetNumDisplayHeads),
  OVERRIDE_ENUM(nvmlVgpuTypeGetResolution),

  /* event part */
  OVERRIDE_ENUM(nvmlEventSetCreate),
  OVERRIDE_ENUM(nvmlEventSetFree),
  OVERRIDE_ENUM(nvmlEventSetWait),
  OVERRIDE_ENUM(nvmlEventSetWait_v2),
  NVML_ENTRY_END
} NVML_OVERRIDE_ENUM_t;

#endif
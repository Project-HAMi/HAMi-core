#ifndef __NVML_OVERRIDE_H__
#define __NVML_OVERRIDE_H__




nvmlReturn_t nvmlInternalGetExportTable(const void **ppExportTable,
                                        void *pExportTableId);


//nvmlReturn_t nvmlDeviceSetTemperatureThreshold(
//    nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp);


//nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance,
//                                              unsigned int *gpuInstanceId);

//nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory);

nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t *pci);

nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v2(
    nvmlDevice_t device,
    nvmlGridLicensableFeatures_t *pGridLicensableFeatures);

nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v3(
    nvmlDevice_t device,
    nvmlGridLicensableFeatures_t *pGridLicensableFeatures);

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device,
                                                     unsigned int *infoCount,
                                                     nvmlProcessInfo_v2_t *infos);

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos);

#endif
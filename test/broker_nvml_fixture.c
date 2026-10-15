#include <stdint.h>

#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

// Loaded by the production NVML pre-init code in the broker mapping test.
nvmlReturn_t nvmlInit_v2(void) {
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    *device = (nvmlDevice_t)(uintptr_t)(index + 1);
    return NVML_SUCCESS;
}

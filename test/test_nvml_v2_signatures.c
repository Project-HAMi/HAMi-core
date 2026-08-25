/*
 * GPU-free regression test for the versioned NVML pass-through wrappers.
 *
 * nvmlDeviceGetFanSpeed_v2 takes a fan index that nvmlDeviceGetFanSpeed does
 * not, and nvmlDeviceGetRetiredPages_v2 takes a timestamps array that
 * nvmlDeviceGetRetiredPages does not.  Both wrappers forward through
 * driver_sym_t, an unprototyped function pointer, and this translation unit
 * never sees NVIDIA's nvml.h, so a wrapper declared with the v1 arity
 * compiles and links with no diagnostic.  The argument the wrapper never
 * declared then reaches the driver as whatever the corresponding argument
 * register happens to hold.
 *
 * The test installs its own nvml_library_entry table and calls the production
 * wrappers through the real NVML prototypes, the way an application does.
 * Recorded pointers are compared and never dereferenced unless they already
 * match, so a wrong-arity build fails cleanly instead of writing through a
 * stray address.
 *
 * LIBCUDA_LOG_LEVEL must be 4.  Below that the LOG_DEBUG branch inside
 * NVML_OVERRIDE_CALL is not taken, nothing clobbers the argument register
 * between the wrapper's entry and the forward, and a v1-arity wrapper can
 * pass the dropped argument through by accident.  The test refuses to run at
 * a lower level rather than report a pass it cannot justify.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/libnvml_hook.h"

/* The production wrappers, declared the way NVIDIA's nvml.h declares them. */
nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan,
                                      unsigned int *speed);
nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device,
                                          nvmlPageRetirementCause_t cause,
                                          unsigned int *pageCount,
                                          unsigned long long *addresses,
                                          unsigned long long *timestamps);

/* Stands in for the loaded libnvidia-ml.so.1 entry table. */
entry_t nvml_library_entry[NVML_ENTRY_END];

#define FAN_SENTINEL 4242u
#define TIMESTAMP_SENTINEL 0x5A5A5A5A5A5A5A5Aull
#define TEST_DEVICE ((nvmlDevice_t)0xD1CEu)

static unsigned int seen_fan;
static unsigned int *seen_speed;
static unsigned int *expect_speed;

static unsigned long long *seen_timestamps;
static unsigned long long *expect_timestamps;
static unsigned long long *seen_addresses;
static unsigned int *seen_page_count;
static nvmlPageRetirementCause_t seen_cause;

static nvmlReturn_t stub_fan_speed_v2(nvmlDevice_t device, unsigned int fan,
                                      unsigned int *speed) {
    (void)device;
    seen_fan = fan;
    seen_speed = speed;
    /* Only written once the pointer is known good, so a broken build cannot
     * turn this test into a wild store. */
    if (speed == expect_speed && speed != NULL) {
        *speed = FAN_SENTINEL;
    }
    return NVML_SUCCESS;
}

static nvmlReturn_t stub_retired_pages_v2(nvmlDevice_t device,
                                          nvmlPageRetirementCause_t cause,
                                          unsigned int *pageCount,
                                          unsigned long long *addresses,
                                          unsigned long long *timestamps) {
    (void)device;
    seen_cause = cause;
    seen_page_count = pageCount;
    seen_addresses = addresses;
    seen_timestamps = timestamps;
    if (timestamps == expect_timestamps && timestamps != NULL) {
        timestamps[0] = TIMESTAMP_SENTINEL;
    }
    return NVML_SUCCESS;
}

static int failures;

static void check(int ok, const char *what) {
    printf("%-4s %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

static int test_fan_speed_v2(void) {
    unsigned int speed = 0;

    expect_speed = &speed;
    seen_fan = 0xEEEEEEEEu;
    seen_speed = NULL;

    printf("nvmlDeviceGetFanSpeed_v2: passing fan=1 speed=%p\n", (void *)&speed);
    nvmlDeviceGetFanSpeed_v2(TEST_DEVICE, 1u, &speed);
    printf("  driver received fan=%u speed=%p\n", seen_fan, (void *)seen_speed);

    check(seen_fan == 1u, "fan index reaches the driver");
    check(seen_speed == &speed, "speed pointer reaches the driver unchanged");
    check(speed == FAN_SENTINEL, "driver's write lands in the caller's variable");
    return 0;
}

static int test_retired_pages_v2(void) {
    unsigned int page_count = 1;
    unsigned long long addresses[1] = {0};
    unsigned long long timestamps[1] = {0};

    expect_timestamps = timestamps;
    seen_timestamps = NULL;
    seen_addresses = NULL;
    seen_page_count = NULL;
    seen_cause = (nvmlPageRetirementCause_t)0;

    printf("nvmlDeviceGetRetiredPages_v2: passing timestamps=%p\n",
           (void *)timestamps);
    nvmlDeviceGetRetiredPages_v2(TEST_DEVICE,
                                 NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
                                 &page_count, addresses, timestamps);
    printf("  driver received timestamps=%p\n", (void *)seen_timestamps);

    check(seen_cause == NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
          "cause reaches the driver");
    check(seen_page_count == &page_count, "pageCount pointer reaches the driver");
    check(seen_addresses == addresses, "addresses pointer reaches the driver");
    check(seen_timestamps == timestamps,
          "timestamps pointer reaches the driver unchanged");
    check(timestamps[0] == TIMESTAMP_SENTINEL,
          "driver's write lands in the caller's timestamps array");
    return 0;
}

int main(void) {
    log_utils_init();
    if (g_log_level < 4) {
        fprintf(stderr,
                "refusing to run: LIBCUDA_LOG_LEVEL=%d, need 4.  Below debug "
                "level the LOG_DEBUG branch is skipped, no call clobbers the "
                "argument register, and a v1-arity wrapper can forward the "
                "dropped argument by accident.\n",
                g_log_level);
        return 2;
    }

    nvml_library_entry[NVML_OVERRIDE_ENUM(nvmlDeviceGetFanSpeed_v2)].fn_ptr =
        (void *)stub_fan_speed_v2;
    nvml_library_entry[NVML_OVERRIDE_ENUM(nvmlDeviceGetRetiredPages_v2)].fn_ptr =
        (void *)stub_retired_pages_v2;

    test_fan_speed_v2();
    test_retired_pages_v2();

    printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

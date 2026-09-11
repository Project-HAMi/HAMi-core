// Regression tests for get_limit_from_env.
//
// The previous implementation computed a pointer to the unit suffix and then
// passed it to strtoul, which overwrote it, so nothing ever checked that the
// text following the number was the suffix that had been matched. "4Gi" and
// "4GB" parsed as 4 bytes, and "-1" wrapped to SIZE_MAX and disabled the limit.
//
// Runs on any host: it touches no CUDA or NVML entry point.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/memory_limit.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define KiB (UINT64_C(1) << 10)
#define MiB (UINT64_C(1) << 20)
#define GiB (UINT64_C(1) << 30)

static int failures;

static void check_key(const char *key, const char *value, size_t expected) {
    if (value == NULL) {
        unsetenv(key);
    } else {
        setenv(key, value, 1);
    }
    size_t got = get_limit_from_env(key);
    if (got != expected) {
        fprintf(stderr, "FAIL %s=%-14s got %zu, want %zu\n",
                key, value ? value : "<unset>", got, expected);
        failures++;
    }
}

static void check(const char *value, size_t expected) {
    check_key(CUDA_DEVICE_MEMORY_LIMIT, value, expected);
}

int main(void) {
    // Values that already worked keep working.
    check("4G", 4 * GiB);
    check("4g", 4 * GiB);
    check("4096M", 4 * GiB);
    check("1024K", MiB);
    check("4194304", 4 * MiB);   // no unit: plain bytes
    check("0", 0);               // 0 means "no limit"

    // What the HAMi device plugin actually writes: fmt.Sprintf("%vm", MiB).
    check("3000m", 3000 * MiB);
    check("4096m", 4 * GiB);
    check_key("CUDA_DEVICE_MEMORY_LIMIT_0", "3000m", 3000 * MiB);

    // The reported bug: Kubernetes-style and long-form units used to yield 4.
    check("4Gi", 4 * GiB);
    check("4GB", 4 * GiB);
    check("4GiB", 4 * GiB);
    check("4gb", 4 * GiB);
    check("4KB", 4 * KiB);
    check("4Mi", 4 * MiB);
    check("4Ki", 4 * KiB);

    // Whitespace at either end and between number and unit.
    check("  4G  ", 4 * GiB);
    check("4 G", 4 * GiB);
    check("4G\n", 4 * GiB);      // value read from a file

    // Fractions are exact and need a unit. "1.5g" is the README's manual
    // style; it used to be silently truncated to 1 GiB.
    check("1.5G", GiB + GiB / 2);
    check("0.5G", GiB / 2);
    check("1.25M", MiB + MiB / 4);
    check("1.0000000005G", GiB);  // below byte precision
    check("4.5", 0);              // fraction of a byte: refused
    check("4.G", 0);
    check("4.5%", 0);

    // Previously silently misread as octal, or dropped a trailing digit.
    check("08G", 8 * GiB);
    check("0x10G", 0);            // hex is no longer accepted

    // Malformed input is treated as unset, not turned into a limit.
    check("-1", 0);               // used to wrap to SIZE_MAX
    check("4GQ", 0);
    check("4G4", 0);
    check("abc", 0);
    check("", 0);
    check("   ", 0);
    check("4Gbi", 0);
    check(NULL, 0);               // unset

    // Overflow is refused, not truncated.
    check("18446744073709551615G", 0);
    check("18446744073709551616", 0);

    // Core-utilization limits share the parser. Percent is accepted and
    // ignored so "50%" keeps meaning 50; 0 is the documented "no limit".
    check_key(CUDA_DEVICE_SM_LIMIT, "50", 50);
    check_key(CUDA_DEVICE_SM_LIMIT, "50%", 50);
    check_key(CUDA_DEVICE_SM_LIMIT, "50 %", 50);
    check_key(CUDA_DEVICE_SM_LIMIT, "0", 0);
    check_key(CUDA_DEVICE_SM_LIMIT, "50%%", 0);

    if (failures != 0) {
        fprintf(stderr, "%d case(s) failed\n", failures);
        return 1;
    }
    printf("all env limit parsing cases passed\n");
    return 0;
}

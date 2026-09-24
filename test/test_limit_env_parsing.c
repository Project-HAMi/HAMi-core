/*
 * GPU-free regression test for get_limit_from_env.
 *
 * The old parser handed its suffix pointer to strtoul, which overwrote it, so
 * nothing checked the text after the number: 4Gi and 4GB read as 4 bytes and
 * -1 wrapped to SIZE_MAX. Refusing matters as much as parsing, because a
 * limit of 0 means no limit at all.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "multiprocess/multiprocess_memory_limit.h"

#define KiB (UINT64_C(1) << 10)
#define MiB (UINT64_C(1) << 20)
#define GiB (UINT64_C(1) << 30)

#define MEM_ENV "CUDA_DEVICE_MEMORY_LIMIT"
#define SM_ENV  "CUDA_DEVICE_SM_LIMIT"

static int failures;

static void expect(const char *env_name, const char *value, uint64_t want) {
    uint64_t got;

    if (setenv(env_name, value, 1) != 0) {
        perror("setenv");
        exit(1);
    }
    got = get_limit_from_env(env_name);
    if (got != want) {
        fprintf(stderr, "%s=\"%s\": got %" PRIu64 ", expected %" PRIu64 "\n",
                env_name, value, got, want);
        failures++;
    }
    unsetenv(env_name);
}

int main(void) {
    log_utils_init();

    /* Spellings that already worked keep their meaning. */
    expect(MEM_ENV, "4G", 4 * GiB);
    expect(MEM_ENV, "4096M", 4 * GiB);
    expect(MEM_ENV, "4K", 4 * KiB);
    expect(MEM_ENV, "4096", 4096);
    expect(MEM_ENV, "0", 0);

    /* The reported ones, all of which used to read as 4 bytes. */
    expect(MEM_ENV, "4Gi", 4 * GiB);
    expect(MEM_ENV, "4GB", 4 * GiB);
    expect(MEM_ENV, "4GiB", 4 * GiB);
    expect(MEM_ENV, "4gb", 4 * GiB);
    expect(MEM_ENV, "  4G ", 4 * GiB);
    expect(MEM_ENV, "4 G", 4 * GiB);
    expect(MEM_ENV, "4Mi", 4 * MiB);
    expect(MEM_ENV, "4Ki", 4 * KiB);

    /* Base 10, so a leading zero is not octal. */
    expect(MEM_ENV, "08G", 8 * GiB);
    expect(MEM_ENV, "07G", 7 * GiB);

    /* A fraction needs a unit to round to. */
    expect(MEM_ENV, "1.5G", GiB + GiB / 2);
    expect(MEM_ENV, "1.5", 0);

    /* Refused. 0 here means no limit, which is why none of these may be
       guessed at instead. */
    expect(MEM_ENV, "-1", 0);
    expect(MEM_ENV, "0x10G", 0);
    expect(MEM_ENV, "abc", 0);
    expect(MEM_ENV, "4T", 0);
    expect(MEM_ENV, "4Gib extra", 0);
    expect(MEM_ENV, "50%", 0);
    expect(MEM_ENV, "", 0);

    /* The SM limit is a percentage, so a unit is refused rather than scaled
       past 100, which the utilization guards read as no limit. */
    expect(SM_ENV, "50", 50);
    expect(SM_ENV, "50%", 50);
    expect(SM_ENV, "50.5", 50);
    expect(SM_ENV, "0", 0);
    expect(SM_ENV, "-1", 0);
    expect(SM_ENV, "50KiB", 0);
    expect(SM_ENV, "50G", 0);
    /* A unit still scales a memory limit. */
    expect(MEM_ENV, "50KiB", 50 * KiB);

    if (failures != 0) {
        fprintf(stderr, "%d limit parsing assertion(s) failed\n", failures);
        return 1;
    }
    puts("limit env parsing tests passed");
    return 0;
}

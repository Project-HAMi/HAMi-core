CUDA_HOME ?= /usr/local/cuda
CFLAGS = -std=gnu11 -O1 -Wall -Wextra
INCLUDES = -I$(CUDA_HOME)/include -I../build/config -I../src
LIBS = -L$(CUDA_HOME)/lib64 -lcuda -lpthread -lm

CORE_TARGETS = bench_init nested_retain warm_holder abi_check
OPTIONAL_TARGETS = phase_probe

# Probe with the exact header path and link flags phase_probe itself uses,
# not just ldconfig -p: the runtime .so.1 can be present via the driver
# package while the dev symlink (-lnvidia-ml) and/or nvml.h (from the CUDA
# toolkit) are still missing, which would make phase_probe a chosen target
# that then fails to link.
HAVE_NVML := $(shell printf '#include <nvml.h>\nint main(void){return 0;}\n' | \
	gcc $(CFLAGS) -x c - $(INCLUDES) $(LIBS) -lnvidia-ml -o /dev/null >/dev/null 2>&1 && echo 1)

TARGETS = $(CORE_TARGETS)
ifeq ($(HAVE_NVML),1)
TARGETS += $(OPTIONAL_TARGETS)
endif

.PHONY: all clean check-cuda-home print-targets

all: $(TARGETS)

# Order-only prerequisite on every binary (not just `all`'s prerequisite
# list), so the preflight also runs under `make -j` and when a target is
# built directly, e.g. `make bench_init`, not just via `make all`.
$(CORE_TARGETS) $(OPTIONAL_TARGETS): | check-cuda-home

# Machine-readable list of targets actually built this run, so callers like
# run_benchmarks.sh can tell whether phase_probe was skipped (no NVML) rather
# than just missing from a stale build.
print-targets:
	@echo "$(TARGETS)"

check-cuda-home:
	@test -f "$(CUDA_HOME)/include/cuda.h" || { \
		echo "error: cuda.h not found under CUDA_HOME=$(CUDA_HOME)"; \
		echo "       set CUDA_HOME to your CUDA toolkit/headers install, e.g.:"; \
		echo "       make CUDA_HOME=/path/to/cuda"; \
		exit 1; \
	}

bench_init: bench_init.c
	gcc $(CFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

phase_probe: phase_probe.c
	gcc $(CFLAGS) $(INCLUDES) $< -o $@ $(LIBS) -lnvidia-ml

nested_retain: nested_retain.c
	gcc $(CFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

warm_holder: warm_holder.c
	gcc $(CFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

abi_check: abi_check.c
	gcc $(CFLAGS) $(INCLUDES) $< -o $@ -lpthread

clean:
	rm -f $(CORE_TARGETS) $(OPTIONAL_TARGETS)

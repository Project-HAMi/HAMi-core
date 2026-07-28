CUDA_HOME ?= /usr/local/cuda
CFLAGS = -std=gnu11 -O1 -Wall -Wextra
INCLUDES = -I$(CUDA_HOME)/include -I../build/config -I../src
LIBS = -L$(CUDA_HOME)/lib64 -lcuda -lpthread -lm

CORE_TARGETS = bench_init nested_retain warm_holder abi_check
OPTIONAL_TARGETS = phase_probe

HAVE_NVML := $(shell ldconfig -p | grep -q libnvidia-ml && echo 1)

TARGETS = $(CORE_TARGETS)
ifeq ($(HAVE_NVML),1)
TARGETS += $(OPTIONAL_TARGETS)
endif

.PHONY: all clean check-cuda-home print-targets

all: check-cuda-home $(TARGETS)

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

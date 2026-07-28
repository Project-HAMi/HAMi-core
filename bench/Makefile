CUDA_HOME ?= /home/muon/cuda-headers
CFLAGS = -O1 -Wall -Wextra
INCLUDES = -I$(CUDA_HOME)/include -I../build/config -I../src
LIBS = -L$(CUDA_HOME)/lib64 -lcuda -lpthread -lm

CORE_TARGETS = bench_init nested_retain warm_holder abi_check
OPTIONAL_TARGETS = phase_probe

TARGETS = $(CORE_TARGETS)
ifeq ($(shell ldconfig -p | grep -q libnvidia-ml && echo 1),1)
TARGETS += $(OPTIONAL_TARGETS)
endif

.PHONY: all clean

all: $(TARGETS)

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

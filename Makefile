.DEFAULT_GOAL := build

current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

build:
	sh ./build.sh
.PHONY: build

build-in-docker:
	docker run -i --rm \
		-v $(current_dir):/libvgpu \
		-w /libvgpu \
		nvidia/cuda:13.3.0-cudnn-devel-ubi8 \
		sh -c "dnf install -y cmake git && \
           git config --global --add safe.directory /libvgpu && \
           rm -rf /libvgpu/build && \
           bash ./build.sh"
.PHONY: build-in-docker

check-cuda-hook-consistency:
	python3 hack/check_cuda_hook_consistency.py
.PHONY: check-cuda-hook-consistency

test_alloc:
	@mkdir -p build/test
	nvcc -o build/test/test_alloc test/main.cu -lcuda -lcudart
.PHONY: test_alloc

test_lock:
	@mkdir -p build/test
	gcc -o build/test/bench_lock test/bench_lock.c src/utils.c -I./src
.PHONY: test_lock

test: build test_alloc test_lock
	@echo "\n=== Running HAMi-core Test & Benchmark Suite ==="
	@total=0; passed=0; \
	run_test() { \
		total=$$((total + 1)); \
		echo "\n--- Test $$total: $$1 ---"; \
		eval "$$2"; \
		if [ $$? -eq 0 ]; then \
			passed=$$((passed + 1)); \
			echo "[✅ PASSED] $$1"; \
		else \
			echo "[❌ FAILED] $$1"; \
		fi; \
	}; \
	run_test "Baseline Allocation (No Preload)" "./build/test/test_alloc"; \
	run_test "HAMi-core Interception Test (4GB Limit)" "LD_PRELOAD=$(current_dir)build/libvgpu.so CUDA_DEVICE_MEMORY_LIMIT=4096m LIBCUDA_LOG_LEVEL=4 ./build/test/test_alloc"; \
	run_test "Lock Contention Benchmark (50 workers)" "./build/test/bench_lock"; \
	run_test "Concurrent CUDA Hook Test (20 processes)" "for i in \$$(seq 1 20); do LD_PRELOAD=$(current_dir)build/libvgpu.so CUDA_DEVICE_MEMORY_LIMIT=4096m ./build/test/test_alloc > /dev/null & done; wait"; \
	echo "\n========================================"; \
	echo "$$total tests executed, $$passed passed"; \
	if [ "$$passed" -eq "$$total" ]; then \
		exit 0; \
	else \
		exit 1; \
	fi
.PHONY: test

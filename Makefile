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

test: build test_alloc
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
	run_test "Lock Contention Benchmark (50 Workers x 10 Iterations)" "/usr/bin/time -v sh -c 'for worker in \$$(seq 1 50); do (for iter in \$$(seq 1 10); do LD_PRELOAD=$(current_dir)build/libvgpu.so CUDA_DEVICE_MEMORY_LIMIT=4096m ./build/test/test_alloc > /dev/null 2>&1; done) & done; wait'"; \
	echo "\n========================================"; \
	echo "$$total tests executed, $$passed passed"; \
	if [ "$$passed" -eq "$$total" ]; then \
		exit 0; \
	else \
		exit 1; \
	fi
.PHONY: test

clean:
	@echo "Cleaning up build directory..."
	rm -rf build/
.PHONY: clean

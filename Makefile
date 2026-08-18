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
	@echo "\n=== Running Baseline Test (No Preload) ==="
	./build/test/test_alloc
	@echo "\n=== Running HAMi-core Interception Test (4GB Limit) ==="
	LD_PRELOAD=$(current_dir)build/libvgpu.so CUDA_DEVICE_MEMORY_LIMIT=4096m LIBCUDA_LOG_LEVEL=4 ./build/test/test_alloc
.PHONY: test

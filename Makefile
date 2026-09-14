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
           bash ./build.sh && \
           echo /usr/local/cuda/lib64/stubs > /etc/ld.so.conf.d/zz-cuda-stubs.conf && \
           ldconfig && \
           ctest --test-dir /libvgpu/build --output-on-failure --no-tests=error"
.PHONY: build-in-docker

check-cuda-hook-consistency:
	python3 hack/check_cuda_hook_consistency.py
.PHONY: check-cuda-hook-consistency

.DEFAULT_GOAL := build

current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

# Derived from dockerfiles/Dockerfile (kept in sync with Project-HAMi/HAMi's NVIDIA_IMAGE).
CUDA_IMAGE ?= $(shell awk '/^FROM / {print $$2; exit}' $(current_dir)dockerfiles/Dockerfile)
DOCKER_IMAGE ?= hami-core:dev

build:
	sh ./build.sh
.PHONY: build

build-in-docker:
	docker run -i --rm \
		-v $(current_dir):/libvgpu \
		-w /libvgpu \
		$(CUDA_IMAGE) \
		sh -c "dnf install -y cmake git && \
           git config --global --add safe.directory /libvgpu && \
           rm -rf /libvgpu/build && \
           bash ./build.sh && \
           echo /usr/local/cuda/lib64/stubs > /etc/ld.so.conf.d/zz-cuda-stubs.conf && \
           ldconfig && \
           ctest --test-dir /libvgpu/build --output-on-failure --no-tests=error"
.PHONY: build-in-docker

docker-build:
	docker build -f $(current_dir)dockerfiles/Dockerfile -t $(DOCKER_IMAGE) $(current_dir)
.PHONY: docker-build

check-cuda-hook-consistency:
	python3 hack/check_cuda_hook_consistency.py
.PHONY: check-cuda-hook-consistency

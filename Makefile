.DEFAULT_GOAL := build

current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

# Keep in sync with dockerfiles/Dockerfile and Project-HAMi/HAMi's NVIDIA_IMAGE.
CUDA_IMAGE ?= nvidia/cuda:13.3.0-cudnn-devel-ubi8@sha256:e5b2b971730b6d0defd6d1bd7697630e0e599c359190cc4351e3032134e7b401

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
           bash ./build.sh"
.PHONY: build-in-docker

check-cuda-hook-consistency:
	python3 hack/check_cuda_hook_consistency.py
.PHONY: check-cuda-hook-consistency

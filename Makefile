.DEFAULT_GOAL := build

current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

build:
	sh ./build.sh
.PHONY: build

build-in-docker:
	docker run -i --rm \
		-v $(current_dir):/libvgpu \
		-w /libvgpu \
		-e DEBIAN_FRONTEND=noninteractive \
		nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04 \
		sh -c "apt-get -y update; apt-get -y install cmake; bash ./build.sh"
.PHONY: build-in-docker

check-cuda-hook-consistency:
	python3 hack/check_cuda_hook_consistency.py
.PHONY: check-cuda-hook-consistency

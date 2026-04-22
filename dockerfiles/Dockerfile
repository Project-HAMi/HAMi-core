FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /libvgpu
COPY . /libvgpu

RUN apt-get update && \
    apt-get install -y cmake git libvulkan-dev && \
    rm -rf /var/lib/apt/lists/*

RUN bash ./build.sh

# Ship Vulkan implicit layer manifest so HAMI_VULKAN_ENABLE=1 activates the layer.
RUN install -D -m 0644 /libvgpu/etc/vulkan/implicit_layer.d/hami.json \
        /etc/vulkan/implicit_layer.d/hami.json

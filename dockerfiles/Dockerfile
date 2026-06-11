FROM nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /libvgpu
COPY . /libvgpu

RUN apt-get update && \
    apt-get install -y cmake git && \
    rm -rf /var/lib/apt/lists/*

RUN bash ./build.sh

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /libvgpu
COPY . /libvgpu

RUN apt-get update && \
    apt-get install -y cmake git && \
    rm -rf /var/lib/apt/lists/*

RUN bash ./build.sh

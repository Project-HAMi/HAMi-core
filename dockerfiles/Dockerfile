FROM nvidia/cuda:12.2.0-devel-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

FROM base AS builder

WORKDIR /libvgpu
COPY . /libvgpu

RUN bash ./build.sh

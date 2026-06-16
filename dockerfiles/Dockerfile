FROM nvidia/cuda:13.3.0-cudnn-devel-ubi8

WORKDIR /libvgpu
COPY . /libvgpu

RUN dnf install -y cmake git && \
    dnf clean all

RUN bash ./build.sh

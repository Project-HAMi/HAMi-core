FROM nvidia/cuda:13.4.3-cudnn-devel-ubi8@sha256:53d7fab8667b6524238d2ff5b03b4be788e7afd7c00e5cd15cc2af8f911d8010

WORKDIR /libvgpu
COPY . /libvgpu

RUN dnf install -y cmake git && \
    dnf clean all

RUN bash ./build.sh

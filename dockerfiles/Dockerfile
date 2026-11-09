FROM nvidia/cuda:13.5.0-cudnn-devel-ubi8@sha256:25ec0bcb3b5bfaf567a105e07268ef685ef81e0a02c79142f4fea3e32d11a7f3

WORKDIR /libvgpu
COPY . /libvgpu

RUN dnf install -y cmake git && \
    dnf clean all

RUN bash ./build.sh

FROM nvidia/cuda:13.4.1-cudnn-devel-ubi8@sha256:366e2fe84d10498e8d379d8d973c1fd4c009dfda33c41bef17d10f01b399130a

WORKDIR /libvgpu
COPY . /libvgpu

RUN dnf install -y cmake git && \
    dnf clean all

RUN bash ./build.sh

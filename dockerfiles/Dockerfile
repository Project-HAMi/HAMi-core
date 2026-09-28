FROM nvidia/cuda:13.3.0-cudnn-devel-ubi8@sha256:e5b2b971730b6d0defd6d1bd7697630e0e599c359190cc4351e3032134e7b401

WORKDIR /libvgpu
COPY . /libvgpu

RUN dnf install -y cmake git && \
    dnf clean all

RUN bash ./build.sh

FROM nvidia/cuda:13.3.1-cudnn-devel-ubi8@sha256:30d5cbaf3661d0b897d9f75d52fdc0494e6a91210356cb0490444cf6aea72e04

WORKDIR /libvgpu
COPY . /libvgpu

RUN dnf install -y cmake git && \
    dnf clean all

RUN bash ./build.sh

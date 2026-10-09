FROM nvidia/cuda:13.4.2-cudnn-devel-ubi8@sha256:7bf81df28de77ac76b1415edfa506645118fd72546842c58b50e544a28d69ec2

WORKDIR /libvgpu
COPY . /libvgpu

RUN dnf install -y cmake git && \
    dnf clean all

RUN bash ./build.sh

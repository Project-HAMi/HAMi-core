FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
COPY . /libvgpu
WORKDIR /libvgpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update; apt-get -y install cmake
RUN bash ./build.sh

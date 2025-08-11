# HAMi-core —— Hook library for CUDA Environments

English | [中文](README_CN.md)

## Introduction

HAMi-core is the in-container gpu resource controller, it has beed adopted by [HAMi](https://github.com/Project-HAMi/HAMi), [volcano](https://github.com/volcano-sh/devices)

<img src="./docs/images/hami-arch.png" width = "600" /> 

## Features

HAMi-core has the following features:
1. Virtualize device meory
2. Limit device utilization by self-implemented time shard
3. Real-time device utilization monitor 

![image](docs/images/sample_nvidia-smi.png)

## Design

HAMi-core operates by Hijacking the API-call between CUDA-Runtime(libcudart.so) and CUDA-Driver(libcuda.so), as the figure below:

<img src="./docs/images/hami-core-position.png" width = "400" />

## Build in Docker

```bash
make build-in-docker
```

## Usage

_CUDA_DEVICE_MEMORY_LIMIT_ indicates the upper limit of device memory (eg 1g,1024m,1048576k,1073741824) 

_CUDA_DEVICE_SM_LIMIT_ indicates the sm utility percentage of each device

```bash
# Add 1GiB memory limit and set max SM utility to 50% for all devices
export LD_PRELOAD=./build/libvgpu.so
export CUDA_DEVICE_MEMORY_LIMIT=1g
export CUDA_DEVICE_SM_LIMIT=50
```

If you run CUDA applications locally, please create the local directory first.

```
mkdir /tmp/vgpulock/
```

```
If you have updated `CUDA_DEVICE_MEMORY_LIMIT` or `CUDA_DEVICE_SM_LIMIT`, please delete the local cache file.

```
rm /tmp/cudevshr.cache
```

## Docker Images

```bash
# Build docker image
docker build . -f=dockerfiles/Dockerfile -t cuda_vmem:tf1.8-cu90

# Configure GPU device and library mounts for container
export DEVICE_MOUNTS="--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidiactl:/dev/nvidiactl"
export LIBRARY_MOUNTS="-v /usr/cuda_files:/usr/cuda_files -v $(which nvidia-smi):/bin/nvidia-smi"

# Run container and check nvidia-smi output
docker run ${LIBRARY_MOUNTS} ${DEVICE_MOUNTS} -it \
    -e CUDA_DEVICE_MEMORY_LIMIT=2g \
    -e LD_PRELOAD=/libvgpu/build/libvgpu.so \
    cuda_vmem:tf1.8-cu90 \
    nvidia-smi
```

After running, you will see nvidia-smi output similar to the following, showing memory limited to 2GiB:

```
...
[HAMI-core Msg(1:140235494377280:libvgpu.c:836)]: Initializing.....
Mon Dec  2 04:38:12 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.107.02             Driver Version: 550.107.02     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        Off |   00000000:03:00.0 Off |                  N/A |
| 30%   36C    P8              7W /  170W |       0MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
[HAMI-core Msg(1:140235494377280:multiprocess_memory_limit.c:497)]: Calling exit handler 1
```

## Log

Use environment variable LIBCUDA_LOG_LEVEL to set the visibility of logs

| LIBCUDA_LOG_LEVEL | description |
| ----------------- | ----------- |
|  0          | errors only |
|  1(default),2          | errors,warnings,messages |
|  3                | infos,errors,warnings,messages |
|  4                | debugs,errors,warnings,messages |

## Test Raw APIs

```bash
./test/test_alloc
```

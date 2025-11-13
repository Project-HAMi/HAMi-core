# softmig —— GPU Resource Controller for SLURM Environments

**softmig** (formerly HAMi-core) is a fork optimized for **Digital Research Alliance Canada (DRAC) / Compute Canada** SLURM environments. It provides software-based GPU memory and compute cycle limiting for oversubscribed GPU partitions.

**Original Project**: [HAMi-core](https://github.com/Project-HAMi/HAMi-core) - in-container GPU resource controller adopted by [HAMi](https://github.com/Project-HAMi/HAMi) and [volcano](https://github.com/volcano-sh/devices)

## Introduction

softmig enables GPU oversubscription on SLURM clusters by:
- **Virtualizing GPU memory**: Restrict memory per job (e.g., 12GB, 24GB slices)
- **Limiting GPU compute cycles**: Restrict SM utilization percentage (e.g., 25%, 50%)
- **Multi-job coordination**: Track and enforce limits across jobs sharing the same GPU

This fork adds:
- SLURM integration (uses `SLURM_TMPDIR`, `SLURM_JOB_ID` for isolation)
- Silent operation (no user-visible logs)
- Structured logging to `/var/log/softmig/`
- Environment-first limit validation
- Optimized for Compute Canada/Alliance clusters

**Note**: This is designed for SLURM job environments, not Docker containers.

<img src="./docs/images/hami-arch.png" width = "600" /> 

## Features

softmig provides the following features:
1. **Virtualize device memory**: Limit GPU memory per job
2. **Limit device utilization**: Control SM utilization percentage via time sharding
3. **Real-time device utilization monitor**: Track GPU usage across multiple jobs

## Design

softmig operates by hijacking API calls between CUDA-Runtime(libcudart.so) and CUDA-Driver(libcuda.so), as shown below:

<img src="./docs/images/hami-core-position.png" width = "400" />

## Building

```bash
# Build with CUDA 11 headers for maximum compatibility (works with CUDA 11, 12, 13)
export CUDA_HOME=/path/to/cuda-11.8
./build.sh

# The library will be built as: build/libsoftmig.so
```

## Usage in SLURM Jobs

softmig is designed to run in SLURM job environments. It automatically uses `SLURM_TMPDIR` for cache and lock files, and logs to `/var/log/softmig/`.

### Environment Variables

- `CUDA_DEVICE_MEMORY_LIMIT`: Upper limit of device memory (e.g., `1g`, `24G`, `1024m`, `1048576k`, `1073741824`)
- `CUDA_DEVICE_SM_LIMIT`: SM utilization percentage (0-100)
- `LD_PRELOAD`: Path to `libsoftmig.so` library

### Basic Test Example

Here's a simple test to verify softmig is working in a SLURM job:

```bash
# In your SLURM job script or interactive session:

# 1. DELETE the cache file FIRST (important when changing limits!)
rm -f ${SLURM_TMPDIR}/cudevshr.cache*

# 2. Set your memory limit
export CUDA_DEVICE_MEMORY_LIMIT=16g

# 3. Load the library (adjust path to your installation)
export LD_PRELOAD=/var/lib/shared/libsoftmig.so
# Or: export LD_PRELOAD=/opt/softmig/lib/libsoftmig.so
# Or if testing from build directory: export LD_PRELOAD=/path/to/HAMi-core/build/libsoftmig.so

# 4. Test with nvidia-smi (should show limited memory)
nvidia-smi
```

**Working Example** (from Compute Canada cluster):
```bash
# Build the library
module load cuda/12.2
rm -rf build && ./build.sh

# In your job or interactive session:
# 1. Delete cache
rm -f ${SLURM_TMPDIR}/cudevshr.cache*

# 2. Set limit
export CUDA_DEVICE_MEMORY_LIMIT=16g

# 3. Load library (from build directory for testing)
export LD_PRELOAD=/project/def-rahimk/rahimk/hami-2/HAMi-core/build/libsoftmig.so

# 4. Test
nvidia-smi
# Should show: 0MiB / 16384MiB (16GB limit)
```

Expected output showing memory limited to 16GB:
```
Wed Nov 12 18:41:53 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40S                    On  |   00000000:4A:00.0 Off |                    0 |
| N/A   27C    P8             31W /  350W |       0MiB /  16384MiB |      0%      Default |
```

### Python/PyTorch Example

```bash
# In your SLURM job:
export LD_PRELOAD=/var/lib/shared/libsoftmig.so
export CUDA_DEVICE_MEMORY_LIMIT=12g
export CUDA_DEVICE_SM_LIMIT=50  # Optional: limit to 50% SM utilization

# Clear cache
rm -f ${SLURM_TMPDIR}/cudevshr.cache*

# Run your Python script
python your_script.py
```

### Important Notes

- **Cache files**: Located in `$SLURM_TMPDIR/cudevshr.cache.*` (auto-cleaned when job ends)
- **Lock files**: Located in `$SLURM_TMPDIR/vgpulock/lock.*` (per-job isolation)
- **Logs**: Written to `/var/log/softmig/{user}_{jobid}_{date}.log` (silent to users)
- **Changing limits**: Always delete cache files before setting new limits

## Logging

Logs are written to `/var/log/softmig/{user}_{jobid}_{arrayid}_{date}.log` and are completely silent to users by default.

Use environment variable `LIBCUDA_LOG_LEVEL` to control log verbosity:

| LIBCUDA_LOG_LEVEL | description |
| ----------------- | ----------- |
|  0 (default)      | errors only (silent operation) |
|  2                | errors, warnings, messages |
|  3                | info, errors, warnings, messages |
|  4                | debug, info, errors, warnings, messages |

To view logs (as admin):
```bash
tail -f /var/log/softmig/*.log
```

## Deployment

For cluster administrators, see [DEPLOYMENT_DRAC.md](docs/DEPLOYMENT_DRAC.md) for complete SLURM integration guide.

## Test Programs

```bash
# After building, test basic functionality
cd build
export LD_PRELOAD=./libsoftmig.so
export CUDA_DEVICE_MEMORY_LIMIT=1g
./test/test_alloc
```

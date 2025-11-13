# softmig Deployment Guide for Digital Research Alliance Canada

**softmig** (formerly HAMi-core) is a fork optimized for Digital Research Alliance Canada (DRAC) / Compute Canada SLURM environments. It provides software-based GPU memory and compute cycle limiting for oversubscribed GPU partitions.

## Overview

softmig enables GPU oversubscription by:
- **Memory Limiting**: Restricts GPU memory per job (e.g., 12GB, 24GB slices on 48GB GPUs)
- **SM Utilization Limiting**: Restricts GPU compute cycles (e.g., 50% SM utilization)
- **Multi-job Coordination**: Tracks and enforces limits across multiple jobs sharing the same GPU

## Installation

### 1. Build the Library

```bash
# Build with CUDA 11 headers for maximum compatibility
export CUDA_HOME=/cvmfs/soft.computecanada.ca/config/gpu/cuda/11.8
cd /path/to/softmig
./build.sh

# Install to shared location (accessible to all compute nodes)
# Option 1: /var/lib/shared (if it exists and is accessible)
sudo mkdir -p /var/lib/shared
sudo cp build/libsoftmig.so /var/lib/shared/
sudo chmod 755 /var/lib/shared/libsoftmig.so

# Option 2: /opt/softmig/lib (alternative)
# sudo mkdir -p /opt/softmig/lib
# sudo cp build/libsoftmig.so /opt/softmig/lib/
# sudo chmod 755 /opt/softmig/lib/libsoftmig.so
```

### 2. Create Log Directory

```bash
# Create log directory (requires root/admin)
sudo mkdir -p /var/log/vgpulogs
sudo chmod 755 /var/log/vgpulogs
# Optional: Set up log rotation
```

## SLURM Configuration

### GRES Configuration

Add to `/etc/slurm/gres.conf`:

```bash
AutoDetect=off

# Physical GPUs (keep existing)
NodeName=rack[01-14]-[01-16] Name=gpu Type=l40s File=/dev/nvidia[0-3]

# Logical GPU slices for softmig (no File= parameter)
NodeName=rack[01-14]-[01-16] Name=gpu Type=l40s.1 Count=4   # Full GPU (1x)
NodeName=rack[01-14]-[01-16] Name=gpu Type=l40s.2 Count=8   # Half GPU (2x sharing)
NodeName=rack[01-14]-[01-16] Name=gpu Type=l40s.4 Count=16  # Quarter GPU (4x sharing)
```

### Partition Configuration

Add to `/etc/slurm/slurm.conf`:

```bash
# Half GPU sliced partitions (2x oversubscription)
PartitionName=gpuhalf_bygpu_b1 Nodes=p1,p2,p3,p4,p5 MaxTime=3:00:00 DefaultTime=1:00:00 \
    DefMemPerNode=500 State=UP OverSubscribe=YES:2 \
    TRESBillingWeights=CPU=643.75,Mem=81.10G,GRES/gpu:l40s.2=5150

PartitionName=gpuhalf_bygpu_b2 Nodes=p2,p3,p4,p5 MaxTime=12:00:00 DefaultTime=1:00:00 \
    DefMemPerNode=500 State=UP OverSubscribe=YES:2 \
    TRESBillingWeights=CPU=643.75,Mem=81.10G,GRES/gpu:l40s.2=5150

# Quarter GPU sliced partitions (4x oversubscription)
PartitionName=gpuquarter_bygpu_b1 Nodes=p1,p2,p3,p4,p5 MaxTime=3:00:00 DefaultTime=1:00:00 \
    DefMemPerNode=500 State=UP OverSubscribe=YES:4 \
    TRESBillingWeights=CPU=643.75,Mem=81.10G,GRES/gpu:l40s.4=2575

# Interactive partitions
PartitionName=gpuhalf_interac Nodes=compute MaxTime=8:00:00 DefaultTime=0:10:00 \
    DefMemPerNode=500 State=UP OverSubscribe=YES:2 \
    TRESBillingWeights=CPU=643.75,Mem=81.10G,GRES/gpu:l40s.2=5150

PartitionName=gpuquarter_interac Nodes=compute MaxTime=8:00:00 DefaultTime=0:10:00 \
    DefMemPerNode=500 State=UP OverSubscribe=YES:4 \
    TRESBillingWeights=CPU=643.75,Mem=81.10G,GRES/gpu:l40s.4=2575
```

### Accounting Configuration

Update `/etc/slurm/slurm.conf`:

```bash
AccountingStorageTRES=gres/gpu,gres/gpu:l40s,gres/gpu:l40s.1,gres/gpu:l40s.2,gres/gpu:l40s.4,cpu,mem
PriorityWeightTRES=CPU=15000,Mem=15000,GRES/gpu=15000,GRES/gpu:l40s=15000,GRES/gpu:l40s.1=15000,GRES/gpu:l40s.2=7500,GRES/gpu:l40s.4=3750
```

## Task Prolog Configuration

Create `/etc/slurm/task_prolog.sh` (or update existing):

```bash
#!/bin/bash

# Existing prolog content (proxy, SSH, cache, etc.)
echo export SLURM_TMPDIR=/tmp
echo export http_proxy="http://squid:3128"
echo export https_proxy="http://squid:3128"
echo export no_proxy="localhost,127.0.0.1"

# ... your existing SSH and cache setup ...

# ===== softmig-core GPU SLICING CONFIGURATION =====
# Configure softmig based on requested GPU slice type
if [[ ! -z "$SLURM_JOB_GRES" ]]; then
    if [[ "$SLURM_JOB_GRES" == *"l40s.2"* ]]; then
        # Half GPU - 24GB memory limit, 50% SM utilization (48GB / 2)
        echo export LD_PRELOAD="/opt/softmig-core/lib/libvgpu.so"
        echo export CUDA_DEVICE_MEMORY_LIMIT="24G"
        echo export CUDA_DEVICE_SM_LIMIT="50"
        echo export SOFTMIG_CORE_ENABLE="1"
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Configured softmig-core for half GPU (24GB, 50% SM)"
    elif [[ "$SLURM_JOB_GRES" == *"l40s.4"* ]]; then
        # Quarter GPU - 12GB memory limit, 25% SM utilization (48GB / 4)
        echo export LD_PRELOAD="/opt/softmig-core/lib/libvgpu.so"
        echo export CUDA_DEVICE_MEMORY_LIMIT="12G"
        echo export CUDA_DEVICE_SM_LIMIT="25"
        echo export SOFTMIG_CORE_ENABLE="1"
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Configured softmig-core for quarter GPU (12GB, 25% SM)"
    else
        # Full GPU or legacy request - no softmig (full access)
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Full GPU, no softmig-core"
    fi
fi
```

## Job Submit Lua Script

Create `/etc/slurm/job_submit.lua` (see example file in `docs/examples/`):

Key features:
- Detects GPU slice type (l40s.2, l40s.4) and routes to appropriate partition
- Validates that users can only request 1 slice of fractional GPUs
- Routes jobs to correct partition based on walltime and GPU type

## Memory Limit Management

### Per-Slice Defaults

| GPU Slice | Memory Limit | SM Limit | Use Case |
|-----------|-------------|----------|----------|
| l40s.1 (full) | 48GB | 100% | Large models, full GPU needed |
| l40s.2 (half) | 24GB | 50% | Medium models, 2x oversubscription |
| l40s.4 (quarter) | 12GB | 25% | Small models, 4x oversubscription |
| l40s.8 (eighth) | 6GB | 12.5% | Very small models, 8x oversubscription |

### Environment Variables

- `CUDA_DEVICE_MEMORY_LIMIT`: Memory limit (e.g., "24G", "12G")
- `CUDA_DEVICE_SM_LIMIT`: SM utilization percentage (0-100)
- `CUDA_DEVICE_MEMORY_LIMIT_0`, `CUDA_DEVICE_MEMORY_LIMIT_1`, etc.: Per-device limits
- `CUDA_DEVICE_SM_LIMIT_0`, `CUDA_DEVICE_SM_LIMIT_1`, etc.: Per-device SM limits
- `LIBCUDA_LOG_LEVEL`: Log verbosity (0=errors only, 4=debug)
- `SOFTMIG_LOG_FILE`: Custom log file path (optional)
- `SOFTMIG_LOCK_FILE`: Custom lock file path (optional)

## User Workflow

### Submitting Jobs

```bash
# Full GPU (existing - no changes)
sbatch --gres=gpu:l40s:1 --time=2:00:00 job.sh
sbatch --gres=gpu:1 --time=10:00:00 job.sh

# Half GPU slice (NEW - 24GB, 50% SM)
sbatch --gres=gpu:l40s.2:1 --time=2:00:00 job.sh

# Quarter GPU slice (NEW - 12GB, 25% SM)
sbatch --gres=gpu:l40s.4:1 --time=5:00:00 job.sh

# Eighth GPU slice (NEW - 6GB, 12.5% SM)
sbatch --gres=gpu:l40s.8:1 --time=5:00:00 job.sh

# Interactive with half GPU (NEW)
salloc --gres=gpu:l40s.2:1 --time=1:00:00
```

### Checking Limits

```bash
# In your job script, check nvidia-smi
nvidia-smi
# Should show limited memory (e.g., 12288MiB for quarter GPU)
```

## Monitoring and Troubleshooting

### Log Files

Logs are written to `/var/log/vgpulogs/` with format:
```
/var/log/vgpulogs/{username}_{jobid}_{arrayid}_{date}.log
```

Example: `/var/log/vgpulogs/rahimk_12345_0_20241112.log`

If `/var/log/vgpulogs/` is not writable, logs fall back to `$SLURM_TMPDIR/softmig_*.log`

### Viewing Logs

```bash
# As admin
tail -f /var/log/vgpulogs/*.log

# In job (if log level enabled)
export LIBCUDA_LOG_LEVEL=2
# Run your application
cat $SLURM_TMPDIR/softmig_*.log
```

### Cache Files

Cache files are stored in `$SLURM_TMPDIR` with format:
```
$SLURM_TMPDIR/cudevshr.cache.{jobid}.gpu{gpu_index}
```

These are automatically cleaned when the job ends.

## Differences from Original HAMi-core

1. **SLURM Integration**: Uses `SLURM_TMPDIR`, `SLURM_JOB_ID` for isolation
2. **Silent Operation**: No user-visible logs (file-only logging)
3. **Structured Logging**: Logs to `/var/log/vgpulogs/` with job info
4. **Environment-First**: Always validates limits from environment, updates cache if mismatch
5. **Renamed**: "softmig" (software MIG) to reflect DRAC optimization
6. **Library Name**: `libsoftmig.so` (instead of `libvgpu.so`)

## Support

For issues or questions, contact your cluster administrator or refer to the main README.md.


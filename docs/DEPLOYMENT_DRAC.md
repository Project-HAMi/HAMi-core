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
# On Digital Research Alliance Canada / Compute Canada (CVMFS)
# Load CUDA module - use CUDA 11 for maximum compatibility (works with CUDA 11, 12, 13)
module load cuda/11.8
# Or: module load cuda/12.2

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

### 2. Configure System-Wide Preload (REQUIRED)

**This is the most secure approach - users cannot disable it.**

```bash
# Add library to /etc/ld.so.preload (one-time setup, requires root)
# This loads the library for ALL processes, but it's passive until limits are configured
echo "/var/lib/shared/libsoftmig.so" | sudo tee -a /etc/ld.so.preload

# OR if using alternative location:
# echo "/opt/softmig/lib/libsoftmig.so" | sudo tee -a /etc/ld.so.preload

# Verify it was added:
cat /etc/ld.so.preload
```

**Why this is safe:**
- The library only activates when limits are configured (config file or env vars)
- If no limits are set, it passes through to real CUDA functions (no overhead)
- Users cannot modify `/etc/ld.so.preload` (requires root)
- More secure than `LD_PRELOAD` because users can't disable it

**To temporarily disable (if needed):**
```bash
# Comment out the line in /etc/ld.so.preload (requires root)
sudo sed -i 's|^/var/lib/shared/libsoftmig.so|#/var/lib/shared/libsoftmig.so|' /etc/ld.so.preload
```

### 3. Create Log and Config Directories

```bash
# Create log directory (requires root/admin)
sudo mkdir -p /var/log/softmig
sudo chmod 755 /var/log/softmig

# Create config directory for secure job config files (requires root/admin)
sudo mkdir -p /var/run/softmig
sudo chown root:root /var/run/softmig
sudo chmod 755 /var/run/softmig
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
NodeName=rack[01-14]-[01-16] Name=gpu Type=l40s.8 Count=32  # Eighth GPU (8x sharing)
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
AccountingStorageTRES=gres/gpu,gres/gpu:l40s,gres/gpu:l40s.1,gres/gpu:l40s.2,gres/gpu:l40s.4,gres/gpu:l40s.8,cpu,mem
PriorityWeightTRES=CPU=15000,Mem=15000,GRES/gpu=15000,GRES/gpu:l40s=15000,GRES/gpu:l40s.1=15000,GRES/gpu:l40s.2=7500,GRES/gpu:l40s.4=3750,GRES/gpu:l40s.8=1875
```

## Task Prolog Configuration

### Task Prolog (Creates Secure Config Files)

**Note:** With `/etc/ld.so.preload`, the library is already loaded system-wide. The prolog only needs to create the config file to activate limits for the job. No `LD_PRELOAD` or wrapper functions are needed.

Create `/etc/slurm/task_prolog.sh` (or update existing):

**Critical**: softmig uses secure config files in `/var/run/softmig/{jobid}_{arrayid}.conf` **instead of environment variables**. 

**Priority**: Config file → Environment variables (if config file doesn't exist)

**Important**: 
- Config files are created **BEFORE** the job starts (in `task_prolog.sh`)
- If a config file exists, environment variables are **ignored** (config file takes priority)
- Users cannot modify config files (admin-only directory `/var/run/softmig/`)
- Config files are deleted **AFTER** the job ends (in `task_epilog.sh`)

**Security Note**: With `/etc/ld.so.preload`, the library is loaded system-wide and users **cannot disable it**. The library is passive (does nothing) until a config file is created, which activates limits for that job. This is the most secure approach.

```bash
#!/bin/bash
# SLURM task_prolog.sh for softmig (Digital Research Alliance Canada)
# This script runs BEFORE each job task and creates secure config files

# Existing prolog content (proxy, SSH, cache, etc.)
echo export SLURM_TMPDIR=/tmp
echo export http_proxy="http://squid:3128"
echo export https_proxy="http://squid:3128"
echo export no_proxy="localhost,127.0.0.1"

# ... your existing SSH and cache setup ...

# ===== softmig GPU SLICING CONFIGURATION =====
# Configure softmig based on requested GPU slice type
# This creates secure config files that users cannot modify
#
# REQUIREMENT: Library must be added to /etc/ld.so.preload (one-time setup, requires root)
#   echo "/var/lib/shared/libsoftmig.so" | sudo tee -a /etc/ld.so.preload
#
# HOW IT WORKS:
# - Library is loaded system-wide via /etc/ld.so.preload (users cannot disable it)
# - Library is passive (does nothing) until a config file is created
# - This prolog creates the config file, which activates the library for this job
# - No LD_PRELOAD or wrapper functions needed - it's automatic and secure

# Create config directory if it doesn't exist (with proper ownership and permissions)
mkdir -p /var/run/softmig
chown root:root /var/run/softmig
chmod 755 /var/run/softmig

if [[ ! -z "$SLURM_JOB_GRES" ]]; then
    # Build config file path: /var/run/softmig/{jobid}_{arrayid}.conf or /var/run/softmig/{jobid}.conf
    CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}"
    if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
        CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.conf"
    else
        CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}.conf"
    fi
    
    if [[ "$SLURM_JOB_GRES" == *"l40s.2"* ]]; then
        # Half GPU - 24GB memory limit, 50% SM utilization (48GB L40S / 2)
        # Create secure config file (users cannot modify this)
        cat > "$CONFIG_FILE" <<EOF
CUDA_DEVICE_MEMORY_LIMIT=24G
CUDA_DEVICE_SM_LIMIT=50
EOF
        chown root:root "$CONFIG_FILE"
        chmod 644 "$CONFIG_FILE"
        # Clear any existing cache
        echo "rm -f \${SLURM_TMPDIR}/cudevshr.cache* 2>/dev/null"
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Configured softmig for half GPU (24GB, 50% SM) via $CONFIG_FILE"

    elif [[ "$SLURM_JOB_GRES" == *"l40s.4"* ]]; then
        # Quarter GPU - 12GB memory limit, 25% SM utilization (48GB L40S / 4)
        cat > "$CONFIG_FILE" <<EOF
CUDA_DEVICE_MEMORY_LIMIT=12G
CUDA_DEVICE_SM_LIMIT=25
EOF
        chown root:root "$CONFIG_FILE"
        chmod 644 "$CONFIG_FILE"
        # Clear any existing cache
        echo "rm -f \${SLURM_TMPDIR}/cudevshr.cache* 2>/dev/null"
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Configured softmig for quarter GPU (12GB, 25% SM) via $CONFIG_FILE"

    elif [[ "$SLURM_JOB_GRES" == *"l40s.8"* ]]; then
        # Eighth GPU - 6GB memory limit, 12.5% SM utilization (48GB L40S / 8)
        cat > "$CONFIG_FILE" <<EOF
CUDA_DEVICE_MEMORY_LIMIT=6G
CUDA_DEVICE_SM_LIMIT=12
EOF
        chown root:root "$CONFIG_FILE"
        chmod 644 "$CONFIG_FILE"
        # Clear any existing cache
        echo "rm -f \${SLURM_TMPDIR}/cudevshr.cache* 2>/dev/null"
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Configured softmig for eighth GPU (6GB, 12% SM) via $CONFIG_FILE"

    else
        # Full GPU or legacy request - no softmig (full access to GPU)
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Full GPU, no softmig"
    fi
fi
```

**See `docs/examples/slurm_task_prolog.sh` for the complete example.**

### Task Epilog (Cleanup)

Create `/etc/slurm/task_epilog.sh` (or update existing) to clean up config files **AFTER** the job ends:

```bash
#!/bin/bash
# SLURM task_epilog.sh for softmig (Digital Research Alliance Canada)
# This script runs AFTER each job task and cleans up softmig config files

# ===== softmig CLEANUP =====
# Delete config file for this job (backup cleanup - exit_handler also does this)
if [[ ! -z "$SLURM_JOB_ID" ]]; then
    CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}"
    if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
        CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.conf"
    else
        CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}.conf"
    fi
    
    if [[ -f "$CONFIG_FILE" ]]; then
        rm -f "$CONFIG_FILE"
        logger -t slurm_task_epilog "Job $SLURM_JOB_ID: Cleaned up softmig config file $CONFIG_FILE"
    fi
fi
```

**See `docs/examples/slurm_task_epilog.sh` for the complete example.**

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

### Configuration Priority

**In SLURM jobs**: Config files take priority over environment variables.

1. **Config file** (`/var/run/softmig/{jobid}_{arrayid}.conf`) - **Created by task_prolog, takes priority**
2. **Environment variables** - Only used if config file doesn't exist (for testing outside SLURM)

**Important**: If a config file exists, environment variables are **ignored**. This ensures users cannot bypass limits by modifying environment variables.

### Environment Variables (Fallback Only)

These are only used if no config file exists (for testing outside SLURM):

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

## Security Considerations

With `/etc/ld.so.preload`, users **cannot disable softmig** because:
- The library is loaded by the dynamic linker before any user code runs
- Users cannot modify `/etc/ld.so.preload` (requires root)
- Even if users unset `LD_PRELOAD`, the library from `/etc/ld.so.preload` is still loaded
- The library only activates when a config file exists (created by prolog), so it's safe for all processes

**Why this is safe:**
- The library is passive (does nothing) when no limits are configured
- For non-GPU jobs or processes without config files, it just passes through to real CUDA functions
- Only GPU slice jobs get config files, which activate the library for those specific jobs

**Monitoring:**
- Regularly check logs to verify softmig is working: `/var/log/softmig/*.log`
- Monitor GPU usage vs. requested slices - if usage exceeds slice limits, investigate
- Check that config files are being created/deleted properly by prolog/epilog

## Monitoring and Troubleshooting

### Log Files

Logs are written to `/var/log/softmig/` with format:
```
/var/log/softmig/{username}_{jobid}_{arrayid}_{date}.log
```

Example: `/var/log/softmig/rahimk_12345_0_20241112.log`

If `/var/log/softmig/` is not writable, logs fall back to `$SLURM_TMPDIR/softmig_*.log`

### Viewing Logs

```bash
# As admin
tail -f /var/log/softmig/*.log

# In job (if log level enabled)
export LIBCUDA_LOG_LEVEL=2
# Run your application
cat $SLURM_TMPDIR/softmig_*.log
```

### Cache Files

Cache files are stored in `$SLURM_TMPDIR` with format:
```
$SLURM_TMPDIR/cudevshr.cache.{jobid}
```

**Important**: Cache files are stored in `SLURM_TMPDIR` only (not regular `/tmp`) for proper job isolation. They are automatically cleaned when the job ends.

**When changing limits**: Always delete cache files before setting new limits:
```bash
rm -f ${SLURM_TMPDIR}/cudevshr.cache*
```

## Differences from Original HAMi-core

1. **SLURM Integration**: Uses `SLURM_TMPDIR`, `SLURM_JOB_ID` for isolation (not regular `/tmp`)
2. **Silent Operation**: No user-visible logs (file-only logging)
3. **Structured Logging**: Logs to `/var/log/softmig/` with job info
4. **Environment-First**: Always validates limits from environment, updates cache if mismatch
5. **Renamed**: "softmig" (software MIG) to reflect DRAC optimization
6. **Library Name**: `libsoftmig.so` (instead of `libvgpu.so`)
7. **SLURM-Only**: Designed for SLURM job environments, not Docker containers

## Support

For issues or questions, contact your cluster administrator or refer to the main README.md.


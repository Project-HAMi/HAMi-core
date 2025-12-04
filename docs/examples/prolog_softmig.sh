#!/bin/bash
# SLURM prolog.sh for softmig
# This script runs before each job and configures softmig
# based on the requested GPU slice type

# ===== softmig GPU SLICING CONFIGURATION =====
# REQUIREMENT: Library must be added to /etc/ld.so.preload (one-time setup, requires root)
#   echo "/var/lib/shared/libsoftmig.so" | sudo tee -a /etc/ld.so.preload
#
# HOW IT WORKS:
# - Library is loaded system-wide via /etc/ld.so.preload (users cannot disable it)
# - Library is passive (does nothing) until a config file is created
# - This prolog creates the config file, which activates the library for this job

set -euo pipefail

# Configuration: Number of shards per GPU (must match job_submit.lua NUM_SHARDS)
NUM_SHARDS_PER_GPU=4

REQ_TRES=$(scontrol show job ${SLURM_JOB_ID} --json | jq ".jobs[0].tres_req_str")

# Check partition name since SLURM_JOB_GRES isn't always set for srun
if [[ "${REQ_TRES:-}" == *"gres/shard"* ]]; then
    # Create config directory with proper permissions
    mkdir -p /var/run/softmig
    chown root:root /var/run/softmig
    chmod 755 /var/run/softmig

    # Build config file path
    CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}.conf"
    
    # Get total GPU memory from first GPU using nvidia-smi (in MB, convert to GB)
    TOTAL_GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -n1 || echo "")
    if [[ -z "$TOTAL_GPU_MEMORY_MB" ]] || ! [[ "$TOTAL_GPU_MEMORY_MB" =~ ^[0-9]+$ ]]; then
        # Fallback to default if nvidia-smi fails or returns invalid value
        TOTAL_GPU_MEMORY_MB=46068 # Replace with default value
    fi
    
    # Remove JSON quotes if present and extract shard count from REQ_TRES
    # Format: gres/shard=count, gres/shard:gpu_name:count, or gres/shard:count (may appear multiple times)
    REQ_TRES_CLEAN=$(echo "$REQ_TRES" | sed 's/^"//;s/"$//')
    SHARD_COUNT=0
    
    # Extract all shard counts and sum them
    # Handle formats: gres/shard=count, gres/shard:gpu_name:count, and gres/shard:count (GPU type is ignored)
    while IFS= read -r shard_spec; do
        # Match gres/shard=count (equals sign format)
        if [[ "$shard_spec" =~ gres/shard=([0-9]+) ]]; then
            count="${BASH_REMATCH[1]}"
            SHARD_COUNT=$((SHARD_COUNT + count))
        fi
    done < <(echo "$REQ_TRES_CLEAN" | grep -oE 'gres/shard=[0-9]+' || true)
    
    # Default to 1 shard if parsing failed
    if [[ $SHARD_COUNT -eq 0 ]]; then
        SHARD_COUNT=1
    fi
    
    # Calculate proportional limits using awk (no bc dependency)
    # Memory: (shard_count / num_shards_per_gpu) * total_memory
    MEMORY_MB=$(awk -v shards="$SHARD_COUNT" -v total_shards="$NUM_SHARDS_PER_GPU" -v total_mem="$TOTAL_GPU_MEMORY_MB" \
        'BEGIN {result = (shards / total_shards) * total_mem; if (result < 0.1) result = 0.1; printf "%.1f", result}')
    
    # SM: (shard_count / num_shards_per_gpu) * 100
    SM_PERCENT=$(awk -v shards="$SHARD_COUNT" -v total_shards="$NUM_SHARDS_PER_GPU" \
        'BEGIN {result = int((shards / total_shards) * 100 + 0.5); if (result < 1) result = 1; printf "%d", result}')
    
    MEMORY_LIMIT="${MEMORY_MB}M"
    
    cat > "$CONFIG_FILE" <<EOFINNER
CUDA_DEVICE_MEMORY_LIMIT=${MEMORY_LIMIT}
CUDA_DEVICE_SM_LIMIT=${SM_PERCENT}
EOFINNER
    chown root:root "$CONFIG_FILE"
    chmod 644 "$CONFIG_FILE"
    rm -f "${SLURM_TMPDIR:-/tmp}/cudevshr.cache"* 2>/dev/null || true
    logger -t slurm_prolog "Job $SLURM_JOB_ID on partition $SLURM_JOB_PARTITION: Configured softmig (${SHARD_COUNT}/${NUM_SHARDS_PER_GPU} shards = ${MEMORY_LIMIT}/${SM_PERCENT}% SM)"
else
    logger -t slurm_prolog "Job $SLURM_JOB_ID on partition ${SLURM_JOB_PARTITION:-unknown}: Full GPU (no softmig)"
fi

exit 0
#!/bin/bash
# SLURM task_prolog.sh for softmig (Digital Research Alliance Canada)
# This script runs before each job task and configures softmig
# based on the requested GPU slice type
#
# Add this section to your existing task_prolog.sh
# Keep your existing proxy, SSH, cache settings, etc. above this section

# ===== softmig GPU SLICING CONFIGURATION =====
# Configure softmig based on requested GPU slice type
# This automatically sets memory and SM utilization limits
# Library location: /var/lib/shared/libsoftmig.so (or /opt/softmig/lib/libsoftmig.so)

# Determine library path (check common locations)
SOFTMIG_LIB=""
if [[ -f "/var/lib/shared/libsoftmig.so" ]]; then
    SOFTMIG_LIB="/var/lib/shared/libsoftmig.so"
elif [[ -f "/opt/softmig/lib/libsoftmig.so" ]]; then
    SOFTMIG_LIB="/opt/softmig/lib/libsoftmig.so"
else
    # Fallback to old location for compatibility
    SOFTMIG_LIB="/opt/softmig/lib/libsoftmig.so"
fi

if [[ ! -z "$SLURM_JOB_GRES" ]]; then
    if [[ "$SLURM_JOB_GRES" == *"l40s.2"* ]]; then
        # Half GPU - 24GB memory limit, 50% SM utilization (48GB L40S / 2)
        echo export LD_PRELOAD="$SOFTMIG_LIB"
        echo export CUDA_DEVICE_MEMORY_LIMIT="24G"
        echo export CUDA_DEVICE_SM_LIMIT="50"
        echo export SOFTMIG_ENABLE="1"
        # Clear any existing cache
        echo "rm -f \${SLURM_TMPDIR}/cudevshr.cache* 2>/dev/null"
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Configured softmig for half GPU (24GB, 50% SM)"
        
    elif [[ "$SLURM_JOB_GRES" == *"l40s.4"* ]]; then
        # Quarter GPU - 12GB memory limit, 25% SM utilization (48GB L40S / 4)
        echo export LD_PRELOAD="$SOFTMIG_LIB"
        echo export CUDA_DEVICE_MEMORY_LIMIT="12G"
        echo export CUDA_DEVICE_SM_LIMIT="25"
        echo export SOFTMIG_ENABLE="1"
        # Clear any existing cache
        echo "rm -f \${SLURM_TMPDIR}/cudevshr.cache* 2>/dev/null"
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Configured softmig for quarter GPU (12GB, 25% SM)"
        
    elif [[ "$SLURM_JOB_GRES" == *"l40s.8"* ]]; then
        # Eighth GPU - 6GB memory limit, 12.5% SM utilization (48GB L40S / 8)
        echo export LD_PRELOAD="$SOFTMIG_LIB"
        echo export CUDA_DEVICE_MEMORY_LIMIT="6G"
        echo export CUDA_DEVICE_SM_LIMIT="12"
        echo export SOFTMIG_ENABLE="1"
        # Clear any existing cache
        echo "rm -f \${SLURM_TMPDIR}/cudevshr.cache* 2>/dev/null"
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Configured softmig for eighth GPU (6GB, 12% SM)"
        
    else
        # Full GPU or legacy request - no softmig (full access to GPU)
        logger -t slurm_task_prolog "Job $SLURM_JOB_ID: Full GPU, no softmig"
    fi
fi

# Note: Logs are completely silent to users by default
# Set LIBCUDA_LOG_LEVEL=2 in job if debugging needed


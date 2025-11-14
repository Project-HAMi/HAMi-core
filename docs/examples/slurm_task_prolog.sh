#!/bin/bash
# SLURM task_prolog.sh for softmig (Digital Research Alliance Canada)
# This script runs before each job task and configures softmig
# based on the requested GPU slice type
#
# Add this section to your existing task_prolog.sh
# Keep your existing proxy, SSH, cache settings, etc. above this section

# ===== softmig GPU SLICING CONFIGURATION =====
# Configure softmig based on requested GPU slice type
# This automatically sets memory and SM utilization limits via secure config file
#
# REQUIREMENT: Library must be added to /etc/ld.so.preload (one-time setup, requires root)
#   echo "/var/lib/shared/libsoftmig.so" | sudo tee -a /etc/ld.so.preload
#   OR: echo "/opt/softmig/lib/libsoftmig.so" | sudo tee -a /etc/ld.so.preload
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

# Note: Logs are completely silent to users by default
# Set LIBCUDA_LOG_LEVEL=2 in job if debugging needed


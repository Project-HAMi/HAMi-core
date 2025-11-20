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

# Create config directory if it doesn't exist (with proper ownership and permissions)
mkdir -p /var/run/softmig
chown root:root /var/run/softmig
chmod 755 /var/run/softmig

# Create log directory if it doesn't exist (with proper ownership and permissions)
# Logs need to be writable by users (library runs as user)
mkdir -p /var/log/softmig
# Use group writable if slurm group exists, otherwise use sticky bit
if getent group slurm >/dev/null 2>&1; then
    chown root:slurm /var/log/softmig
    chmod 775 /var/log/softmig  # drwxrwxr-x (group writable - more secure)
else
    chown root:root /var/log/softmig
    chmod 1777 /var/log/softmig  # drwxrwxrwt (sticky bit - prevents deletion of others' files)
fi

if [[ ! -z "$SLURM_JOB_GRES" ]]; then
    # Build config file path: /var/run/softmig/{jobid}_{arrayid}.conf or /var/run/softmig/{jobid}.conf
    CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}"
    if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
        CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.conf"
    else
        CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}.conf"
    fi
    
    if [[ "$SLURM_JOB_GRES" == *"gpu_slice:titan.2"* ]] || [[ "$SLURM_JOB_GRES" == *"titan.2"* ]]; then
        # Half GPU - 12GB memory limit, 50% SM utilization (adjust if your Titans are 12GB - use 6G)
        cat > "$CONFIG_FILE" <<EOF
CUDA_DEVICE_MEMORY_LIMIT=12G
CUDA_DEVICE_SM_LIMIT=50
EOF
        chown root:root "$CONFIG_FILE"
        chmod 644 "$CONFIG_FILE"
        
        # Clear any existing cache
        rm -f ${SLURM_TMPDIR}/cudevshr.cache* 2>/dev/null
        logger -t slurm_prolog "Job $SLURM_JOB_ID: Configured softmig for half GPU (12GB, 50% SM) via $CONFIG_FILE"
    else
        # Full GPU or legacy request - no softmig (full access to GPU)
        logger -t slurm_prolog "Job $SLURM_JOB_ID: Full GPU, no softmig"
    fi
fi


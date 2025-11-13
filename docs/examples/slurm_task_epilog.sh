#!/bin/bash
# SLURM task_epilog.sh for softmig (Digital Research Alliance Canada)
# This script runs after each job task and cleans up softmig config files
#
# Add this section to your existing task_epilog.sh

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


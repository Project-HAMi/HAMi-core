#!/bin/bash
# SoftMig wrapper script - ensures LD_PRELOAD is always set for GPU slice jobs
# This script prevents users from disabling softmig by unsetting LD_PRELOAD
#
# INSTALLATION:
#   sudo cp softmig_wrapper.sh /usr/local/bin/
#   sudo chmod 755 /usr/local/bin/softmig_wrapper.sh
#
# USAGE OPTIONS:
#
# Option 1: Source in job script (recommended)
#   source /usr/local/bin/softmig_wrapper.sh
#   python your_script.py
#
# Option 2: Use as wrapper for specific commands
#   /usr/local/bin/softmig_wrapper.sh python your_script.py
#
# Option 3: Add to job script template (enforced for all jobs)
#   Add "source /usr/local/bin/softmig_wrapper.sh" to your job script template

# Determine library path
SOFTMIG_LIB=""
if [[ -f "/var/lib/shared/libsoftmig.so" ]]; then
    SOFTMIG_LIB="/var/lib/shared/libsoftmig.so"
elif [[ -f "/opt/softmig/lib/libsoftmig.so" ]]; then
    SOFTMIG_LIB="/opt/softmig/lib/libsoftmig.so"
fi

# Function to ensure softmig is in LD_PRELOAD
ensure_softmig_loaded() {
    if [[ -z "$SOFTMIG_LIB" ]]; then
        return 1  # Library not found
    fi
    
    # Check if this is a GPU slice job (has config file)
    if [[ ! -z "$SLURM_JOB_ID" ]]; then
        CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}"
        if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
            CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.conf"
        else
            CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}.conf"
        fi
        
        # If config file exists, we must have softmig loaded
        if [[ -f "$CONFIG_FILE" ]]; then
            if [[ -z "$LD_PRELOAD" ]] || [[ "$LD_PRELOAD" != *"libsoftmig.so"* ]]; then
                # Re-set LD_PRELOAD with softmig first
                if [[ -z "$LD_PRELOAD" ]]; then
                    export LD_PRELOAD="$SOFTMIG_LIB"
                else
                    # Remove softmig if present, then prepend it
                    LD_PRELOAD_CLEANED=$(echo "$LD_PRELOAD" | sed "s|$SOFTMIG_LIB:||g" | sed "s|:$SOFTMIG_LIB||g" | sed "s|$SOFTMIG_LIB||g")
                    export LD_PRELOAD="$SOFTMIG_LIB:$LD_PRELOAD_CLEANED"
                fi
            fi
        fi
    fi
}

# If script is executed directly (not sourced), run ensure_softmig_loaded and execute command
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly (wrapper mode)
    ensure_softmig_loaded
    # Execute all arguments as a command
    exec "$@"
else
    # Script is being sourced (function mode)
    ensure_softmig_loaded
    # Export function so it can be called in job scripts
    export -f ensure_softmig_loaded
fi


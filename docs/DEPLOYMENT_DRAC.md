# softmig Deployment Guide

> **Note**: This document has been consolidated into the main [README.md](../README.md). 
> 
> For complete deployment instructions, including installation, SLURM configuration, 
> monitoring, and troubleshooting, please see the **"Deployment for Cluster Administrators"** 
> section in [README.md](../README.md).

## Quick Reference

- **Installation**: Use `docs/examples/install_softmig.sh` or see README.md
- **SLURM Configuration**: See README.md "Deployment for Cluster Administrators" section
- **Prolog/Epilog Scripts**: See `docs/examples/prolog_softmig.sh` and `docs/examples/epilog_softmig.sh`
- **Job Submit Plugin** (optional but recommended): See `docs/examples/job_submit_softmig.lua` for GPU slice validation and translation
- **Permissions**: All documented in README.md

## Key Points

- **CUDA 12+ required** (CUDA 11 does not work)
- **System-wide preload**: `/etc/ld.so.preload` must be set to `644` (rw-r--r--)
- **Library permissions**: `/var/lib/shared/libsoftmig.so` must be `644` (readable by all)
- **Log directory**: `775` (group writable) with `slurm` group, or `1777` (sticky bit)
- **Config directory**: `755` (readable by all, writable only by root)
- **Config files**: `/var/run/softmig/{jobid}.conf` or `/var/run/softmig/{jobid}_{arrayid}.conf` (created by prolog, deleted by epilog)
- **Passive mode**: Library operates in passive mode when no config file or environment variables are set (safe for system-wide preload)

For detailed information, see [README.md](../README.md).

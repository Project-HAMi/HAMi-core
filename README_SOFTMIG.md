# softmig - Software GPU Slicing for Digital Research Alliance Canada

**softmig** is a fork of [HAMi-core](https://github.com/Project-HAMi/HAMi-core) optimized for Digital Research Alliance Canada (DRAC) / Compute Canada SLURM environments.

## What is softmig?

**softmig** = **Soft**ware **MIG** (Multi-Instance GPU)

Like NVIDIA's hardware MIG (available on A100/H100), softmig provides software-based GPU slicing for any GPU, enabling:
- **GPU Memory Slicing**: Divide GPU memory among multiple jobs (e.g., 12GB, 24GB slices)
- **GPU Compute Slicing**: Limit SM utilization per job (e.g., 25%, 50% of GPU cycles)
- **Oversubscription**: Run 2-4 jobs per GPU safely

## Key Features for DRAC

✅ **SLURM Integration**: Uses `SLURM_TMPDIR`, `SLURM_JOB_ID` for proper isolation  
✅ **Silent Operation**: No user-visible logs (file-only logging)  
✅ **Structured Logging**: Logs to `/var/log/vgpulogs/{user}_{jobid}_{date}.log`  
✅ **Environment-First**: Always validates limits from environment variables  
✅ **Multi-CUDA Support**: Works with CUDA 11, 12, 13 (build with CUDA 11 headers)  
✅ **SM Utilization Limiting**: Controls GPU compute cycles, not just memory  

## Quick Start

### For Cluster Administrators

See [DEPLOYMENT_DRAC.md](docs/DEPLOYMENT_DRAC.md) for complete deployment guide.

### For Users

```bash
# Request half GPU slice (24GB, 50% SM)
sbatch --gres=gpu:l40s.2:1 --time=2:00:00 job.sh

# Request quarter GPU slice (12GB, 25% SM)
sbatch --gres=gpu:l40s.4:1 --time=5:00:00 job.sh

# Full GPU (no limits)
sbatch --gres=gpu:l40s:1 --time=2:00:00 job.sh
```

## Differences from Original HAMi-core

| Feature | HAMi-core | softmig |
|---------|-----------|--------------|
| Target Environment | Docker/Kubernetes | SLURM clusters |
| Logging | `/tmp/vgpulog` | `/var/log/vgpulogs/{user}_{jobid}_{date}.log` |
| Cache Location | `/tmp/cudevshr.cache` | `$SLURM_TMPDIR/cudevshr.cache.{jobid}` |
| User Visibility | Logs to stderr | Completely silent |
| Limit Validation | Cache-first | Environment-first |
| Isolation | Process-based | Job-based (SLURM_JOB_ID) |

## Documentation

- **[DEPLOYMENT_DRAC.md](docs/DEPLOYMENT_DRAC.md)**: Complete deployment guide for DRAC clusters
- **[Example Configs](docs/examples/)**: SLURM configuration examples
  - `slurm_job_submit.lua`: Job routing and validation
  - `slurm_task_prolog.sh`: Automatic configuration
  - `slurm_gres.conf`: GRES definitions
  - `slurm_partitions.conf`: Partition configurations

## Environment Variables

- `CUDA_DEVICE_MEMORY_LIMIT`: Memory limit (e.g., "24G", "12G")
- `CUDA_DEVICE_SM_LIMIT`: SM utilization percentage (0-100)
- `LIBCUDA_LOG_LEVEL`: Log verbosity (0=errors only, 4=debug)
- `SOFTMIG_LOG_FILE`: Custom log file path (optional)
- `SOFTMIG_LOCK_FILE`: Custom lock file path (optional)

## Building

```bash
# Build with CUDA 11 for maximum compatibility (works with CUDA 11, 12, 13)
export CUDA_HOME=/path/to/cuda-11.8
./build.sh

# Install to system location
sudo mkdir -p /var/lib/shared
sudo cp build/libsoftmig.so /var/lib/shared/
# Or alternative location:
# sudo mkdir -p /opt/softmig/lib
# sudo cp build/libsoftmig.so /opt/softmig/lib/
```

## License

Same as original HAMi-core project.

## Acknowledgments

Based on [HAMi-core](https://github.com/Project-HAMi/HAMi-core) by Project-HAMi, optimized for Digital Research Alliance Canada / Compute Canada infrastructure.


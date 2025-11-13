# SoftMig - Software GPU Slicing for SLURM Clusters

**SoftMig** is a fork of [HAMi-core](https://github.com/Project-HAMi/HAMi-core) optimized for **Digital Research Alliance Canada (DRAC) / Compute Canada** SLURM environments. It provides software-based GPU memory and compute cycle limiting for oversubscribed GPU partitions.

Like NVIDIA's hardware MIG, SoftMig enables software-based GPU slicing for any GPU:
- **GPU Memory Slicing**: Divide GPU memory among multiple jobs (e.g., 12GB, 24GB slices on 48GB GPUs)
- **GPU Compute Slicing**: Limit SM utilization per job (e.g., 25%, 50% of GPU cycles)
- **Oversubscription**: Run 2-8 jobs per GPU safely

## Building

### For Digital Research Alliance Canada / Compute Canada (CVMFS)

```bash
# Load CUDA module from CVMFS
module load cuda/11.8  # Recommended for compatibility (works with CUDA 11, 12, 13)
# Or: module load cuda/12.2

# Build the library
./build.sh

# Install (as admin)
sudo mkdir -p /var/lib/shared /var/log/softmig /var/run/softmig
sudo cp build/libsoftmig.so /var/lib/shared/
sudo chmod 755 /var/lib/shared/libsoftmig.so /var/log/softmig /var/run/softmig
```

### For Other Systems

```bash
export CUDA_HOME=/path/to/cuda-11.8
./build.sh
sudo mkdir -p /var/lib/shared /var/log/softmig /var/run/softmig
sudo cp build/libsoftmig.so /var/lib/shared/
```

## Usage

### Configuration

**In SLURM jobs**: SoftMig reads limits from secure config files in `/var/run/softmig/{jobid}_{arrayid}.conf` (created by `task_prolog.sh`). Users cannot modify these files.

**Outside SLURM (testing)**: SoftMig automatically falls back to environment variables if no config file exists.

### Environment Variables (for testing)

- `CUDA_DEVICE_MEMORY_LIMIT`: Memory limit (e.g., `16g`, `24G`)
- `CUDA_DEVICE_SM_LIMIT`: SM utilization percentage (0-100)
- `LD_PRELOAD`: Path to `libsoftmig.so`
- `LIBCUDA_LOG_LEVEL`: Log verbosity (0=errors only, 4=debug, default 0)

### Quick Test

```bash
# 1. Delete cache (important when changing limits!)
rm -f ${SLURM_TMPDIR}/cudevshr.cache*

# 2. Set limit (for testing - production uses config files)
export CUDA_DEVICE_MEMORY_LIMIT=16g

# 3. Load library
export LD_PRELOAD=/var/lib/shared/libsoftmig.so

# 4. Test
nvidia-smi  # Should show: 0MiB / 16384MiB (16GB limit)
```

### For SLURM Users

Once deployed, simply request GPU slices:

```bash
# Half GPU slice (24GB, 50% SM)
sbatch --gres=gpu:l40s.2:1 --time=2:00:00 job.sh

# Quarter GPU slice (12GB, 25% SM)
sbatch --gres=gpu:l40s.4:1 --time=5:00:00 job.sh

# Eighth GPU slice (6GB, 12% SM)
sbatch --gres=gpu:l40s.8:1 --time=1:00:00 job.sh

# Full GPU (no limits)
sbatch --gres=gpu:l40s:1 --time=2:00:00 job.sh
```

Limits are automatically configured by `task_prolog.sh` based on the requested GPU slice type.

## Memory Limits by GPU Slice

| Slice Type | Memory | SM Limit | Oversubscription |
|------------|--------|----------|------------------|
| l40s.1 (full) | 48GB | 100% | 1x |
| l40s.2 (half) | 24GB | 50% | 2x |
| l40s.4 (quarter) | 12GB | 25% | 4x |
| l40s.8 (eighth) | 6GB | 12% | 8x |

## File Locations

- **Cache files**: `$SLURM_TMPDIR/cudevshr.cache.*` (auto-cleaned when job ends)
- **Lock files**: `$SLURM_TMPDIR/vgpulock/lock.*` (per-job isolation)
- **Config files**: `/var/run/softmig/{jobid}_{arrayid}.conf` (created by task_prolog, deleted on exit)
- **Logs**: `/var/log/softmig/{user}_{jobid}_{arrayid}_{date}.log` (silent to users)

Logs fall back to `$SLURM_TMPDIR/softmig_{jobid}.log` if `/var/log/softmig` is not writable.

## Logging

Logs are written to `/var/log/softmig/{user}_{jobid}_{arrayid}_{date}.log` and are completely silent to users by default.

Use `LIBCUDA_LOG_LEVEL` to control verbosity:
- `0` (default): errors only
- `2`: errors, warnings, messages
- `3`: info, errors, warnings, messages
- `4`: debug, info, errors, warnings, messages

View logs (as admin): `tail -f /var/log/softmig/*.log`

## Deployment for Cluster Administrators

For complete deployment instructions, see **[docs/DEPLOYMENT_DRAC.md](docs/DEPLOYMENT_DRAC.md)**.

### Quick Setup

1. **Build and Install** (see Building section above)

2. **SLURM Configuration**:
   - Update `gres.conf` with GPU slice definitions
   - Update `slurm.conf` with new partitions
   - Create/update `task_prolog.sh` (creates secure config files)
   - Create/update `task_epilog.sh` (cleanup)
   - Create/update `job_submit.lua` (job routing and validation)

3. **Example Configs**: See `docs/examples/` for complete examples

## Important Notes

- **Changing limits**: Always delete cache files before setting new limits
- **Config files**: In SLURM jobs, limits come from secure config files (users cannot modify)
- **Cache files**: Auto-cleaned when job ends (SLURM_TMPDIR is job-specific)
- **Multi-CUDA**: Build with CUDA 11 headers for compatibility with CUDA 11, 12, 13

## License

Same as original [HAMi-core](https://github.com/Project-HAMi/HAMi-core) project.

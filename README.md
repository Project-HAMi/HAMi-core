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
# NOTE: CUDA 12+ required (CUDA 11 does not work)
module load cuda/12.2  # Recommended
# Or: module load cuda/13.0

# Build the library
./build.sh

# Install (as admin)
sudo mkdir -p /var/lib/shared /var/log/softmig /var/run/softmig
sudo cp build/libsoftmig.so /var/lib/shared/
sudo chmod 755 /var/lib/shared/libsoftmig.so
sudo chown root:root /var/log/softmig /var/run/softmig
sudo chmod 755 /var/log/softmig /var/run/softmig

# Configure system-wide preload (REQUIRED for production - users cannot disable it)
echo "/var/lib/shared/libsoftmig.so" | sudo tee -a /etc/ld.so.preload
```

### For Other Systems

```bash
# NOTE: CUDA 12+ required (CUDA 11 does not work)
export CUDA_HOME=/path/to/cuda-12.2
# Or: export CUDA_HOME=/path/to/cuda-13.0
./build.sh
sudo mkdir -p /var/lib/shared /var/log/softmig /var/run/softmig
sudo cp build/libsoftmig.so /var/lib/shared/
```

## Usage

### Configuration

**In SLURM jobs**: SoftMig reads limits from secure config files in `/var/run/softmig/{jobid}_{arrayid}.conf` (created by `prolog.sh`). Users cannot modify these files.

**Outside SLURM (testing)**: SoftMig automatically falls back to environment variables if no config file exists.

### Environment Variables (for testing)

- `CUDA_DEVICE_MEMORY_LIMIT`: Memory limit (e.g., `16g`, `24G`)
- `CUDA_DEVICE_SM_LIMIT`: SM utilization percentage (0-100). Limits GPU compute capacity by throttling kernel launches. Only monitors device 0 (intentional - fractional GPU jobs only get 1 GPU).
- `LD_PRELOAD`: Path to `libsoftmig.so` (for testing only - production uses `/etc/ld.so.preload`)
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

Limits are automatically configured by `prolog.sh` based on the requested GPU slice type.

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
- **Config files**: `/var/run/softmig/{jobid}_{arrayid}.conf` (created by prolog.sh, deleted by epilog.sh)
- **Logs**: `/var/log/softmig/{jobid}_{arrayid}.log` or `/var/log/softmig/{jobid}.log` (silent to users)

Logs fall back to `$SLURM_TMPDIR/softmig_{jobid}.log` if `/var/log/softmig` is not writable.

## Logging

Logs are written to `/var/log/softmig/{jobid}_{arrayid}.log` (or `/var/log/softmig/{jobid}.log` for non-array jobs) and are completely silent to users by default.

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
   - Update `slurm.conf` to add `Prolog` and `Epilog` paths (see `docs/slurm.conf.example`)
   - Create/update `prolog.sh` (creates secure config files) - see `docs/examples/prolog_softmig.sh`
   - Create/update `epilog.sh` (cleanup) - see `docs/examples/epilog_softmig.sh`

3. **Example Scripts**: See `docs/examples/` for prolog/epilog scripts

## Important Notes

- **Changing limits**: Always delete cache files before setting new limits
- **Config files**: In SLURM jobs, limits come from secure config files (users cannot modify)
- **Cache files**: Auto-cleaned when job ends (SLURM_TMPDIR is job-specific)
- **CUDA Version**: **CUDA 12+ required** (tested with CUDA 12.2 and 13.0). CUDA 11 does not work.
- **SM Limiting**: GPU compute utilization limiting works via kernel launch throttling. Only monitors device 0 (intentional - fractional GPU jobs only get 1 GPU). See [docs/GPU_LIMITER_EXPLANATION.md](docs/GPU_LIMITER_EXPLANATION.md) for details.

## License

Same as original [HAMi-core](https://github.com/Project-HAMi/HAMi-core) project.

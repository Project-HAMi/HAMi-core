# SoftMig - Software GPU Slicing for SLURM Clusters

**SoftMig** is a fork of [HAMi-core](https://github.com/Project-HAMi/HAMi-core) optimized for **Digital Research Alliance Canada (DRAC) / Compute Canada** SLURM environments. It provides software-based GPU memory and compute cycle limiting for oversubscribed GPU partitions.

Like NVIDIA's hardware MIG, SoftMig enables software-based GPU slicing for any GPU:
- **GPU Memory Slicing**: Divide GPU memory among multiple jobs (e.g., 12GB, 24GB slices on 48GB GPUs)
- **GPU Compute Slicing**: Limit SM utilization per job (e.g., 25%, 50% of GPU cycles)
- **Oversubscription**: Run 2-8 jobs per GPU safely
- **SLURM Integration**: Uses `SLURM_TMPDIR` for per-job isolation (cache files, locks) - no shared `/tmp` conflicts

## SoftMig vs. NVIDIA Hardware MIG

| Feature | SoftMig (Software MIG) | NVIDIA Hardware MIG |
|---------|------------------------|---------------------|
| **GPU Resource Usage** | Uses 100% of GPU resources (no overhead) | Loses ~5-10% of GPU resources to MIG overhead |
| **Dynamic Configuration** | ✅ Dynamic, on-the-fly changes via SLURM prolog/epilog | ❌ Requires draining SLURM node and rebooting to change MIG mode |
| **Isolation** | ⚠️ Software-based (process-level) - crashes can affect other jobs | ✅ Hardware-based isolation - crashes isolated to MIG instance |
| **GPU Compatibility** | ✅ Works on any NVIDIA GPU (Pascal, Volta, Ampere, Ada, Hopper) | ❌ Only works on A100, H100, and newer datacenter GPUs |
| **Setup Complexity** | ✅ Simple: install library, configure SLURM | ⚠️ Requires GPU driver support, MIG mode configuration |
| **Performance Overhead** | ⚠️ Minimal (~1-2% from kernel launch throttling) | ✅ No software overhead (hardware-native) |
| **Multi-Instance Support** | ✅ Unlimited slices (memory/compute limited) | ⚠️ Limited by GPU architecture (max 7 instances on A100) |

**When to Use SoftMig:**
- Need dynamic GPU slicing without node downtime
- Want to use 100% of GPU resources
- Using GPUs that don't support hardware MIG (L40S, RTX, etc.)
- Need flexible, on-demand slice sizing

**When to Use Hardware MIG:**
- Need maximum isolation (security, fault tolerance)
- Using A100/H100 datacenter GPUs
- Can tolerate node downtime for configuration changes
- Want zero software overhead

## Differences from Original HAMi-core

SoftMig is optimized for SLURM cluster environments with the following key improvements:

| Feature | Original HAMi-core | SoftMig |
|---------|-------------------|---------|
| **Temporary Files** | Uses `/tmp` (shared across all jobs/users) | ✅ Uses `$SLURM_TMPDIR` (per-job isolation) |
| **Cache Files** | `/tmp/cudevshr.cache` (shared, can conflict) | ✅ `$SLURM_TMPDIR/cudevshr.cache.{jobid}` (job-specific) |
| **Lock Files** | `/tmp/vgpulock/` (shared) | ✅ `$SLURM_TMPDIR/vgpulock/` (job-specific) |
| **Configuration** | Environment variables only | ✅ Secure config files (`/var/run/softmig/{jobid}.conf`) + env vars fallback |
| **Logging** | stderr (visible to users) | ✅ File-only logging (`/var/log/softmig/`) - silent to users |
| **Log File Names** | Process ID based | ✅ Job ID based (`{jobid}_{arrayid}.log`) |
| **Library Loading** | `LD_PRELOAD` (users can disable) | ✅ `/etc/ld.so.preload` (users cannot disable) |
| **SLURM Integration** | Manual setup | ✅ Automated via `prolog.sh`/`epilog.sh` |
| **Multi-CUDA Support** | CUDA 11+ | ✅ CUDA 12+ (CUDA 11 does not work) |
| **Library Name** | `libvgpu.so` | ✅ `libsoftmig.so` |

**Key Benefits:**
- ✅ **Job Isolation**: Each SLURM job gets its own cache/lock files in `SLURM_TMPDIR` (no conflicts)
- ✅ **Security**: Config files in `/var/run/softmig/` prevent users from modifying limits
- ✅ **Silent Operation**: No user-visible logs (file-only logging)
- ✅ **Enforcement**: System-wide preload ensures users cannot disable the library
- ✅ **Auto-cleanup**: Cache files automatically cleaned when job ends (SLURM_TMPDIR is job-specific)

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
sudo chmod 644 /var/lib/shared/libsoftmig.so  # rw-r--r-- (readable by all)
sudo chown root:root /var/lib/shared/libsoftmig.so

# Log directory: group writable (775) if slurm group exists, otherwise 1777 (sticky bit)
if getent group slurm >/dev/null 2>&1; then
    sudo chown root:slurm /var/log/softmig
    sudo chmod 775 /var/log/softmig  # drwxrwxr-x (group writable)
else
    sudo chown root:root /var/log/softmig
    sudo chmod 1777 /var/log/softmig  # drwxrwxrwt (sticky bit)
fi

# Config directory (readable by all, writable only by root)
sudo chown root:root /var/run/softmig
sudo chmod 755 /var/run/softmig  # drwxr-xr-x

# Configure system-wide preload (REQUIRED for production - users cannot disable it)
echo "/var/lib/shared/libsoftmig.so" | sudo tee -a /etc/ld.so.preload
sudo chmod 644 /etc/ld.so.preload  # rw-r--r-- (readable by all)
sudo chown root:root /etc/ld.so.preload
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

### Installation

Use the automated installation script (recommended):
```bash
# As root
sudo ./docs/examples/install_softmig.sh /path/to/build/libsoftmig.so
```

Or install manually (see Building section above for full steps).

**Important Permissions:**
- `/var/lib/shared/libsoftmig.so`: `644` (rw-r--r--) - readable by all
- `/var/lib/shared/`: `755` (drwxr-xr-x) - readable/executable by all
- `/var/log/softmig/`: `775` (drwxrwxr-x) with `slurm` group, or `1777` (drwxrwxrwt) with sticky bit
- `/var/run/softmig/`: `755` (drwxr-xr-x) - readable by all, writable only by root
- `/etc/ld.so.preload`: `644` (rw-r--r--) - readable by all

### SLURM Configuration

1. **Update `slurm.conf`**:
   ```bash
   Prolog=/etc/slurm/prolog.sh
   Epilog=/etc/slurm/epilog.sh
   ```
   See `docs/slurm.conf.example` for minimal configuration.

2. **Create/update `prolog.sh`**:
   - Creates secure config files in `/var/run/softmig/{jobid}_{arrayid}.conf`
   - See `docs/examples/prolog_softmig.sh` for complete example
   - Configures limits based on requested GPU slice type (l40s.2, l40s.4, etc.)

3. **Create/update `epilog.sh`**:
   - Cleans up config files after job ends
   - See `docs/examples/epilog_softmig.sh` for complete example

### How It Works

**System-Wide Preload (`/etc/ld.so.preload`):**
- Library is loaded for ALL processes (users cannot disable it)
- Library is passive (does nothing) until a config file is created
- Prolog creates config file → activates limits for that job
- Epilog deletes config file → deactivates limits when job ends

**Config File Priority:**
1. Config file (`/var/run/softmig/{jobid}_{arrayid}.conf`) - **Takes priority** (created by prolog)
2. Environment variables - Only used if config file doesn't exist (for testing)

**Security:**
- Users cannot modify config files (admin-only directory)
- Users cannot disable library (system-wide preload)
- Library only activates when config file exists (safe for all processes)

### Memory and SM Limits by GPU Slice

| GPU Slice | Memory Limit | SM Limit | Oversubscription | Use Case |
|-----------|-------------|----------|------------------|----------|
| l40s.1 (full) | 48GB | 100% | 1x | Large models, full GPU needed |
| l40s.2 (half) | 24GB | 50% | 2x | Medium models, 2x oversubscription |
| l40s.4 (quarter) | 12GB | 25% | 4x | Small models, 4x oversubscription |
| l40s.8 (eighth) | 6GB | 12% | 8x | Very small models, 8x oversubscription |

**SM Limiting:** GPU compute utilization limiting works via kernel launch throttling. Only monitors device 0 (intentional - fractional GPU jobs only get 1 GPU). See [docs/GPU_LIMITER_EXPLANATION.md](docs/GPU_LIMITER_EXPLANATION.md) for technical details.

### Monitoring and Troubleshooting

**Log Files:**
- Location: `/var/log/softmig/{jobid}_{arrayid}.log` or `/var/log/softmig/{jobid}.log`
- View: `tail -f /var/log/softmig/*.log` (as admin)
- Silent to users by default (set `LIBCUDA_LOG_LEVEL=2` in job for debugging)

**Cache Files:**
- Location: `$SLURM_TMPDIR/cudevshr.cache.*` (auto-cleaned when job ends)
- **Important**: Delete cache files when changing limits: `rm -f ${SLURM_TMPDIR}/cudevshr.cache*`

**Verification:**
- Check that config files are created: `ls -l /var/run/softmig/`
- Check that library is loaded: `cat /etc/ld.so.preload`
- Test in job: `nvidia-smi` should show limited memory

## Important Notes

- **SLURM_TMPDIR Integration**: All temporary files (cache, locks) use `$SLURM_TMPDIR` for per-job isolation. This prevents conflicts between concurrent jobs and ensures automatic cleanup when jobs end.
- **Changing limits**: Always delete cache files before setting new limits: `rm -f ${SLURM_TMPDIR}/cudevshr.cache*`
- **Config files**: In SLURM jobs, limits come from secure config files (users cannot modify)
- **Cache files**: Auto-cleaned when job ends (SLURM_TMPDIR is job-specific, unlike original HAMi-core which used shared `/tmp`)
- **CUDA Version**: **CUDA 12+ required** (tested with CUDA 12.2 and 13.0). CUDA 11 does not work.
- **SM Limiting**: GPU compute utilization limiting works via kernel launch throttling. Only monitors device 0 (intentional - fractional GPU jobs only get 1 GPU). See [docs/GPU_LIMITER_EXPLANATION.md](docs/GPU_LIMITER_EXPLANATION.md) for details.

## License

Same as original [HAMi-core](https://github.com/Project-HAMi/HAMi-core) project.

# Changes for Digital Research Alliance Canada (DRAC)

This document summarizes all changes made to HAMi-core to create softmig for DRAC/Compute Canada SLURM environments.

> **Note**: For general usage and deployment, see [README.md](README.md). This document is for developers and administrators who need to understand the technical changes.

## Summary

**softmig** is a fork of HAMi-core optimized for Digital Research Alliance Canada SLURM clusters. It enables GPU oversubscription through software-based memory and compute cycle limiting.

## Key Changes

### 1. Logging System Overhaul

**File**: `src/include/log_utils.h`

**Changes**:
- ✅ All logs now go to file only (no stderr output)
- ✅ Logs written to `/var/log/softmig/{jobid}.log` or `/var/log/softmig/{jobid}_{arrayid}.log` (matches config file naming)
- ✅ Falls back to `$SLURM_TMPDIR/softmig_{jobid}.log` (or `$SLURM_TMPDIR/softmig_{jobid}_{arrayid}.log` for array jobs) if `/var/log` not writable (SLURM_TMPDIR only, not regular /tmp)
- ✅ Outside SLURM jobs: logs use `/var/log/softmig/pid{pid}.log`
- ✅ Completely silent to users (no visible messages)
- ✅ Renamed from "HAMI-core" to "softmig" in log messages
- ✅ Library renamed from `libvgpu.so` to `libsoftmig.so`
- ✅ Supports `SOFTMIG_LOG_FILE` environment variable for custom log path

**Impact**: Users never see log messages, admins can monitor via log files.

### 2. SLURM-Aware Cache and Lock Files

**Files**: 
- `src/multiprocess/multiprocess_memory_limit.c`
- `src/utils.c`

**Changes**:
- ✅ Cache files use `$SLURM_TMPDIR/cudevshr.cache.{jobid}` or `$SLURM_TMPDIR/cudevshr.cache.{jobid}.{arrayid}` for array jobs
- ✅ Per-job isolation prevents cross-job interference
- ✅ Auto-cleaned when job ends (SLURM_TMPDIR is job-specific)
- ✅ Only uses `SLURM_TMPDIR` (not regular `/tmp`) for proper job isolation
- ✅ Cache file cleanup: `prolog_softmig.sh` removes old cache files on job start

**Impact**: Proper isolation between jobs, no cache conflicts.

### 3. Environment-First Limit Validation

**File**: `src/multiprocess/multiprocess_memory_limit.c`

**Changes**:
- ✅ Limit inconsistencies now update cache (not just log error)
- ✅ Environment variables are source of truth
- ✅ Cache is updated if limits don't match environment
- ✅ Changed limit inconsistency from `LOG_ERROR` to `LOG_DEBUG` (silent)
- ✅ Same for SM limit inconsistencies

**Impact**: Users can't bypass limits by deleting cache - it recreates with env limits.

### 4. Improved dlsym Resolution

**File**: `src/libsoftmig.c`, `src/nvml/hook.c`

**Changes**:
- ✅ Multiple fallback methods for getting real `dlsym`
- ✅ Works with CVMFS CUDA (Compute Canada)
- ✅ Uses weak linking for `_dl_sym` (no undefined symbol errors)
- ✅ Tries `dlvsym` with multiple GLIBC versions
- ✅ Better error handling (doesn't break basic commands)

**Impact**: Works reliably on Compute Canada systems with CVMFS CUDA.

### 5. SM Utilization Limiting

**Status**: Already implemented, verified working

**Features**:
- ✅ Limits GPU compute cycles via `CUDA_DEVICE_SM_LIMIT`
- ✅ Works alongside memory limiting
- ✅ Per-device SM limits supported
- ✅ Automatic rate limiting based on utilization

**Usage**: Set `CUDA_DEVICE_SM_LIMIT=50` for 50% SM utilization.

### 6. Secure Config File System

**Files**: 
- `src/multiprocess/config_file.c` (new)
- `src/multiprocess/multiprocess_memory_limit.c`
- `docs/examples/slurm_task_prolog.sh`
- `docs/examples/slurm_task_epilog.sh` (new)

**Changes**:
- ✅ Reads limits from secure config files in `/var/run/softmig/{jobid}.conf` or `/var/run/softmig/{jobid}_{arrayid}.conf` for array jobs
- ✅ Users cannot modify config files (admin-only directory `/var/run/softmig/`)
- ✅ Falls back to environment variables if config file doesn't exist (for non-SLURM testing)
- ✅ Config files automatically deleted on job exit (via `epilog_softmig.sh` and exit handler)
- ✅ Prolog (`prolog_softmig.sh`) creates config files based on GPU slice requests
- ✅ Epilog (`epilog_softmig.sh`) cleans up config files
- ✅ **Passive mode**: Library operates in passive mode when no config file or environment variables are set (safe for system-wide preload)

**Impact**: Prevents users from modifying limits by changing environment variables. Limits are enforced via secure config files.

### 7. Documentation and Examples

**New Files**:
- ✅ `docs/DEPLOYMENT_DRAC.md`: Complete deployment guide
- ✅ `docs/examples/job_submit_softmig.lua`: Validates and translates GPU slice requests (e.g., `l40s.2:1` → `gres/shard:l40s:2`)
- ✅ `docs/examples/prolog_softmig.sh`: Automatic configuration with config files (creates `/var/run/softmig/{jobid}.conf`)
- ✅ `docs/examples/epilog_softmig.sh`: Config file cleanup
- ✅ `docs/examples/install_softmig.sh`: Automated installation script

## Configuration Summary

### Memory Limits by GPU Slice

**Default Configuration**: 4 shards per GPU (configurable in `prolog_softmig.sh` and `job_submit_softmig.lua`)

| Slice Type | Shard Count | Memory (48GB GPU) | SM Limit | Oversubscription | Use Case |
|------------|-------------|-------------------|----------|------------------|----------|
| l40s (full) | N/A | 48GB | 100% | 1x | Large models |
| l40s.2 (half) | 2 shards | 24GB | 50% | 2x | Medium models |
| l40s.4 (quarter) | 1 shard | 12GB | 25% | 4x | Small models |

### Environment Variables

- `CUDA_DEVICE_MEMORY_LIMIT`: Memory limit (e.g., "24G")
- `CUDA_DEVICE_SM_LIMIT`: SM utilization (0-100)
- `LIBCUDA_LOG_LEVEL`: Log verbosity (0-4, default 0=silent)
- `SOFTMIG_LOG_FILE`: Custom log path
- `SOFTMIG_LOCK_FILE`: Custom lock path

## Deployment Checklist

- [ ] Build library with CUDA 12+ (CUDA 11 does not work)
- [ ] Install to `/var/lib/shared/libsoftmig.so` (or `/opt/softmig/lib/libsoftmig.so`)
- [ ] Create `/var/log/softmig/` directory
- [ ] Create `/var/run/softmig/` directory (for secure config files)
- [ ] Update `slurm.conf` with new partitions
- [ ] Update `gres.conf` with GPU slices
- [ ] Create/update `task_prolog.sh` (creates secure config files)
- [ ] Create/update `task_epilog.sh` (cleans up config files)
- [ ] Create/update `job_submit.lua`
- [ ] Test with sample jobs
- [ ] Document for users

## Testing

Test with:
```bash
# Half GPU
sbatch --gres=gpu:l40s.2:1 --time=1:00:00 test_job.sh

# Quarter GPU  
sbatch --gres=gpu:l40s.4:1 --time=1:00:00 test_job.sh
```

Verify:
- nvidia-smi shows limited memory
- No user-visible log messages
- Logs appear in `/var/log/softmig/`
- Cache files in `$SLURM_TMPDIR`

## Notes

- Library name changed to `libsoftmig.so`
- Project is renamed to "softmig" in documentation
- Library is renamed to `libsoftmig.so`
- All changes are backward compatible with original HAMi-core usage
- **CUDA 12+ required** (tested with CUDA 12.2 and 13.0). CUDA 11 does not work.


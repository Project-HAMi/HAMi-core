# GPU Compute Utilization Limiter (SM Limit) - How It Works

## Overview

The GPU compute utilization limiter (`CUDA_DEVICE_SM_LIMIT`) restricts how much of the GPU's compute capacity (SM utilization) a job can use. This is separate from memory limiting and works by throttling kernel launches.

## How It Works

### 1. **Token-Based Rate Limiting**

The system uses a **token bucket** algorithm:

- **`g_total_cuda_cores`**: Total available "CUDA cores" (tokens)
  - Calculated as: `SM_count × max_threads_per_SM × FACTOR` (where FACTOR=32)
  - Example: 48 SMs × 2048 threads × 32 = ~3.1M tokens

- **`g_cur_cuda_cores`**: Current available tokens (decremented on kernel launch)

### 2. **Kernel Launch Throttling**

When `cuLaunchKernel` is called:

```c
// In src/cuda/memory.c:545
CUresult cuLaunchKernel(...) {
    // Calculate kernel "size" (grid dimensions)
    rate_limiter(gridDimX * gridDimY * gridDimZ, blockDimX * blockDimY * blockDimZ);
    // Then launch the actual kernel
    CUDA_OVERRIDE_CALL(...);
}
```

The `rate_limiter()` function:
1. Calculates kernel size = `gridDimX × gridDimY × gridDimZ`
2. Atomically decrements `g_cur_cuda_cores` by kernel size
3. **If tokens are negative (exhausted), it sleeps** until tokens are available
4. This throttles kernel launches to stay within the limit

### 3. **Dynamic Adjustment (Utilization Watcher)**

A background thread (`utilization_watcher`) runs every 120ms:

1. **Measures actual SM utilization** using NVML:
   - `nvmlDeviceGetProcessUtilization()` - gets SM utilization % per process
   - Sums utilization for all processes in the job

2. **Compares to target limit**:
   - Target: `CUDA_DEVICE_SM_LIMIT` (e.g., 50% = 50)
   - Current: Actual measured utilization (0-100%)

3. **Adjusts token allocation**:
   - If utilization < limit: Increase tokens (allow more kernels)
   - If utilization > limit: Decrease tokens (throttle more)
   - Uses `delta()` function to calculate adjustment

4. **Token replenishment**:
   - Tokens are added back via `change_token()` based on utilization feedback

### 4. **The Delta Function**

```c
long delta(int up_limit, int user_current, long share) {
    // Calculate how much to adjust tokens
    int utilization_diff = abs(up_limit - user_current);
    long increment = (SM_num² × max_threads × utilization_diff) / 2560;
    
    if (user_current <= up_limit) {
        share = share + increment;  // Increase tokens
    } else {
        share = share - increment;   // Decrease tokens
    }
    return share;
}
```

## Example Flow

**Setting**: `CUDA_DEVICE_SM_LIMIT=50` (50% utilization)

1. **Initialization**:
   - `g_total_cuda_cores` = 3,145,728 (for 48 SM GPU)
   - `g_cur_cuda_cores` = 3,145,728 (full tokens available)

2. **Kernel Launch**:
   - User launches kernel with grid(1000, 1, 1)
   - `rate_limiter(1000, ...)` called
   - Decrements `g_cur_cuda_cores` by 1000
   - Kernel launches

3. **Utilization Watcher** (every 120ms):
   - Measures: User is using 60% SM utilization
   - Target is 50%, so utilization is too high
   - Calculates negative delta, reduces tokens
   - Future kernel launches will be throttled more

4. **Throttling**:
   - If `g_cur_cuda_cores` goes negative, `rate_limiter()` sleeps
   - Kernel launch is delayed until tokens are replenished
   - This reduces effective utilization

## Key Code Locations

- **Kernel hook**: `src/cuda/memory.c:545` - `cuLaunchKernel()`
- **Rate limiter**: `src/multiprocess/multiprocess_utilization_watcher.c:34` - `rate_limiter()`
- **Watcher thread**: `src/multiprocess/multiprocess_utilization_watcher.c:178` - `utilization_watcher()`
- **Initialization**: `src/multiprocess/multiprocess_utilization_watcher.c:213` - `init_utilization_watcher()`

## Limitations

1. **Only affects kernel launches** - Memory transfers, etc. are not throttled
2. **Coarse-grained** - Works at kernel launch level, not instruction level
3. **Measurement delay** - 120ms polling interval means adjustments lag actual usage
4. **Device 0 only** - Currently only monitors device 0 (`userutil[0]`). This is **intentional by design**:
   - Fractional GPU jobs (the ones that need SM limiting) only get a single GPU slice
   - Full GPU jobs typically have `SM_LIMIT=100` (no limit) or `0` (disabled), so they don't need monitoring
   - If you need multi-GPU SM limiting in the future, the code can be extended to track per-device token pools

## Configuration

Set in config file or environment:
```
CUDA_DEVICE_SM_LIMIT=50  # 50% utilization
```

- `0` or `>=100`: No limit (disabled)
- `1-99`: Percentage of SM utilization allowed


# HAMi-core Performance Optimizations

## Background

The HAMi-core CUDA library hijacking layer was introducing ~23% overhead
on training workloads. Profiling identified the overhead in per-call work
performed by intercepted CUDA functions — not in the actual resource
limiting logic, but in bookkeeping (logging, status checks, shared memory
reads) that executed on every CUDA API call regardless of whether limiting
was active.

This document describes the P0 (critical) and P1 (high-impact)
optimizations applied to reduce this overhead.

---

## Commit 1 — [P0] Cache log level (`log_utils.h`, `utils.c`, `libvgpu.c`)

### Problem
Every `LOG_DEBUG`, `LOG_INFO`, `LOG_WARN`, and `LOG_MSG` macro called
`getenv("LIBCUDA_LOG_LEVEL")` and `atoi()` on every invocation.
`getenv()` performs a linear scan of the environment block. These macros
appear in every intercepted CUDA function, so this overhead accumulated
across thousands of calls per second.

### Fix
- Added `int g_log_level` global variable (default: 2 = warn level).
- Added `log_utils_init()` function that reads `LIBCUDA_LOG_LEVEL` once.
- Rewrote all `LOG_*` macros to check `g_log_level` (a single integer
  comparison) instead of calling `getenv()` + `atoi()`.
- `log_utils_init()` is called from `preInit()` in `libvgpu.c`.

### Behavior preserved
- Default log level (env unset) remains 2 (WARN/MSG/ERROR), matching the
  original behavior where `LOG_WARN`/`LOG_MSG` logged when env was NULL.
- `LOG_ERROR` remains unconditional (always emitted).
- `LIBCUDA_LOG_LEVEL` env var still controls log level; it is simply
  read once at library load time.

### Testing
- Set `LIBCUDA_LOG_LEVEL=4` and verify DEBUG output appears.
- Set `LIBCUDA_LOG_LEVEL=0` and verify only ERROR output appears.
- Unset `LIBCUDA_LOG_LEVEL` and verify WARN/MSG/ERROR output appears
  (same as before).

---

## Commit 2 — [P0] Use cached slot in `wait_status_self()` (`multiprocess_memory_limit.c`)

### Problem
`wait_status_self()` is called via the `ENSURE_RUNNING()` macro on every
kernel launch and memory operation. It performed a linear scan through
all process slots (up to 1024 entries), comparing PIDs via `getpid()` to
find the current process's status field. This was O(n) per call.

### Fix
- Use the already-cached `region_info.my_slot` pointer (set during
  `init_proc_slot_withlock()` at startup) for O(1) direct access.
- Fall back to linear scan only if `my_slot` is NULL (defensive, should
  not happen after initialization).
- Used proper `atomic_load_explicit` with `memory_order_acquire` for
  reading shared-memory fields.

### Behavior preserved
- Returns the same values as before: 1 if status matches, 0 if not, -1
  if process not found.
- The slow path (linear scan) is identical to the original logic.

### Testing
- Run any CUDA workload with HAMi — `ENSURE_RUNNING()` is exercised on
  every kernel launch and memory allocation.

---

## Commit 3 — [P1] Optimize `pre_launch_kernel()` (`multiprocess_memory_limit.c`)

### Problem
`pre_launch_kernel()` runs on every `cuLaunchKernel` call. It was:
1. Calling `time(NULL)` — a syscall into the kernel.
2. Always acquiring `pthread_mutex_lock` even when the timestamp had not
   changed (recording interval is 1 second, kernels fire thousands/sec).

### Fix
- Replaced `time(NULL)` with `clock_gettime(CLOCK_REALTIME_COARSE)`,
  which is served from the Linux vDSO (no syscall). Resolution is ~1-4ms,
  which is irrelevant for a 1-second recording interval. Uses
  `CLOCK_REALTIME_COARSE` (not `MONOTONIC`) to preserve epoch-time
  semantics for `dump_shrreg` and other consumers.
- Added double-checked locking: check the timestamp *before* acquiring
  the mutex. The fast path (>99.99% of calls) becomes a single memory
  read + integer comparison. The mutex is only taken when an update is
  actually needed (~once per second).

### Correctness notes
- The unlocked read of `region_info.last_kernel_time` is safe: `uint64_t`
  reads are atomic on x86-64 and aarch64 (aligned). A torn read would at
  worst cause one unnecessary mutex acquisition, not incorrect behavior.
- The atomic CAS update to the shared region is unchanged.

### Testing
- Run `dump_shrreg` tool while a CUDA workload is active — verify
  `last_kernel_time` still updates correctly (once per second).
- Run a kernel-intensive workload and compare throughput with/without
  this change.

---

## Commit 4 — [P1] Optimize `rate_limiter()` (`multiprocess_utilization_watcher.c`)

### Problem
`rate_limiter()` runs on every kernel launch when `pidfound==1`. Before
reaching the actual rate-limiting CAS loop, it performed:
- `get_recent_kernel()` — shared memory read
- `set_recent_kernel(2)` — shared memory write (always writing 2, which
  was already the value — a no-op that dirtied a cross-process cache line)
- `get_current_device_sm_limit(0)` — called twice (redundant)
- `get_utilization_switch()` — shared memory read

That is 3 shared memory reads + 1 write + 2 `ensure_initialized()` calls
on every kernel launch, even when rate limiting was inactive.

### Fix
- Cache `sm_limit` and `utilization_switch` in static locals during
  `init_utilization_watcher()`. These values are set at container startup
  and do not change at runtime.
- Fast-exit check uses cached locals: when limiting is inactive
  (sm_limit >= 100 or == 0), `rate_limiter` returns after a single
  branch on a local variable.
- Removed `set_recent_kernel(2)` — eliminated the shared memory write.
- Removed the duplicate `get_current_device_sm_limit(0)` call.
- Reduced `sleep(1)` to `usleep(1000)` in the defensive `recent_kernel`
  guard (currently unreachable but safer if triggered externally).
- The CAS spin loop and 10ms `nanosleep` backoff are **unchanged**,
  preserving correct rate-limiting when 0 < sm_limit < 100.

### Behavior preserved
- When SM limiting is active (0 < sm_limit < 100), the token-bucket
  mechanism works identically — the only difference is reaching the CAS
  loop faster (fewer shared memory reads before it).
- The utilization watcher thread is still created under the same
  conditions.

### Testing
- With `CUDA_DEVICE_SM_LIMIT=50`: verify SM utilization is capped as
  before. Run a compute-heavy workload and confirm utilization stays
  near the configured limit.
- With `CUDA_DEVICE_SM_LIMIT=100` (or unset): verify no rate limiting
  occurs and kernel throughput matches native (no HAMi) baseline.

---

## Commit 5 — [P1] Remove dead `cuDeviceGetCount` in `oom_check()` (`allocator.c`)

### Problem
`oom_check()` called `cuDeviceGetCount()` on every memory allocation,
storing the result in `count1` — which was never read. This was a wasted
CUDA driver API call on every allocation.

### Fix
Removed the dead `cuDeviceGetCount` call and the unused `count1`
variable. The function only needs the specific device ID passed via the
`dev` parameter, not the total device count.

### Testing
- Run memory allocation tests (`test_alloc`, `test_runtime_alloc`, etc.)
  to verify OOM checking still works correctly.

---

## Expected Impact

| Change | Per-call overhead removed | Frequency |
|--------|--------------------------|-----------|
| Cached log level | `getenv()` + `atoi()` per LOG macro | Every CUDA call |
| Cached `my_slot` in `wait_status_self` | O(n) linear scan of process slots | Every kernel launch + memory op |
| vDSO clock + double-checked lock | `time()` syscall + mutex lock/unlock | Every kernel launch |
| Cached rate_limiter limits | 3 shared mem reads + 1 write | Every kernel launch |
| Remove dead `cuDeviceGetCount` | 1 driver API call | Every memory allocation |

Combined, these changes should reduce the hijacking overhead from ~23%
to under 5% for typical training workloads.

## How to benchmark

```bash
# Baseline (no HAMi):
python ../semantic-id-recsys/semantic-id-training/test_hami_slowdown.py

# With HAMi (original):
LD_PRELOAD=/path/to/original/libvgpu.so \
  python ../semantic-id-recsys/semantic-id-training/test_hami_slowdown.py

# With HAMi (optimized):
LD_PRELOAD=/path/to/optimized/libvgpu.so \
  python ../semantic-id-recsys/semantic-id-training/test_hami_slowdown.py
```

Compare wall-clock time and throughput (samples/sec) across the three runs.

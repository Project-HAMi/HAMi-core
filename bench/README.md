# HAMi-core Initialization Benchmark Suite

Concurrent CUDA initialization performance benchmarks for HAMi-core's device sharing layer. These tools measure the cost of multiprocess cuInit() and primary context operations under various conditions, supporting the analysis in issue #1662.

## Tools

### bench_init
Measures concurrent cuInit() latency distribution across N worker processes, each spawned via fork+exec through `libvgpu.so` via `LD_PRELOAD`.

```bash
./bench_init N
```

Outputs: per-process initialization time in milliseconds, percentile distribution (p50, p90, p99, max), and aggregated cost. Requires a live GPU and CUDA libraries.

### phase_probe
Breaks down the cost of `set_task_pid()` (host PID detection via NVML snapshot diffing) into phases:
- nvmlInit
- Device snapshot enumeration
- Primary context retain/release cycles

Useful for identifying which phase dominates the initialization overhead.

```bash
./phase_probe
```

### nested_retain
Tests whether `cuDevicePrimaryCtxRetain` cost is proportional to refcount depth or constant once a context exists. Answers: is it cheap to hold a context across set_task_pid() calls?

```bash
./nested_retain
```

### warm_holder
Keeps a GPU context alive in a background process. Used by other tools to test whether context probe cost changes when a context is already held vs. cold.

```bash
./warm_holder [primary|non-primary]
```

### abi_check
Verifies that `struct shared_region_t` layout (sizeof, offsetof for key fields) is byte-identical to the C side. Used to validate that Go-side mirror struct changes don't break the wire format.

```bash
./abi_check
```

## Building

```bash
cd bench
make
```

Requires:
- CUDA headers and libraries (e.g., via `CUDA_HOME`)
- C99 compiler
- `libvgpu.so` built in the parent directory

## Reproducing Issue #1662 Findings

Run the full benchmark suite:

```bash
cd bench
./run_benchmarks.sh
```

This executes:
1. `phase_probe` — identify bottlenecks
2. `warm_holder primary &` — start context holder
3. `bench_init 2 4 8 16 32 64 128` — measure across concurrency levels
4. Kill warm_holder

Expected results:
- ~61.5 ms per-process serialized time (N=1)
- Linear scaling to ~128 processes (each adds ~61.5 ms)
- Phase breakdown: ~80% in cuDevicePrimaryCtx{Retain,Release}, ~20% in NVML
- Nested retain cost: ~1 µs (amortizable)

## Context

These benchmarks were developed to quantify the initialization lock contention identified in #1662. The fix (moving from polling to semaphore-based synchronization) is already merged, but these tools remain useful for regression testing and understanding concurrent device-sharing behavior.

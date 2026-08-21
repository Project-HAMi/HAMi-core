# Init contention benchmark (issue #1662)

`bench_init_contention.c` measures how long HAMi-core initialization takes when
many processes initialize CUDA at the same time. It relates to
[issue #1662](https://github.com/Project-HAMi/HAMi/issues/1662).

It does not change any library behavior. It is an ordinary consumer of the CUDA
driver API, loaded together with `libvgpu.so` through `LD_PRELOAD`, the same way
a real workload process is.

## What it measures

- per-process `cuInit()` latency (min, p50, mean, p95, p99, max)
- total wall time for all processes to finish initialization
- CPU time per process and time spent blocked (wall minus CPU)
- voluntary context switches per process, a proxy for blocking on the shared
  `sem_postinit` semaphore, and involuntary context switches, a proxy for CPU
  preemption when many processes start at once

## Why `cuInit()`

In HAMi-core, `cuInit()` ends by calling `ensure_post_init()`, which runs
`postInit()` once per process. `postInit()` takes the shared memory semaphore
`sem_postinit` through `lock_postinit()` and calls `set_task_pid()` before
releasing it. `set_task_pid()` runs `nvmlInit()`, scans running processes, and
retains a primary CUDA context. That semaphore serializes `set_task_pid()`
across every process on the node, so timing `cuInit()` captures the contention.

## Build

The file is picked up automatically by `test/CMakeLists.txt`.

```sh
./build.sh
```

## Run

Needs an NVIDIA GPU and the built `libvgpu.so`.

```sh
LD_PRELOAD=./build/libvgpu.so CUDA_VISIBLE_DEVICES=0 \
    ./build/test/benchmark/bench_init_contention 128
```

- first argument: number of processes (default 64)
- second argument: optional label printed in the CSV line

To sweep the process count and collect the CSV lines:

```sh
for n in 1 8 32 64 128 256; do
    LD_PRELOAD=./build/libvgpu.so CUDA_VISIBLE_DEVICES=0 \
        ./build/test/benchmark/bench_init_contention "$n" sweep | grep '^CSV,'
done
```

CSV columns: `label, processes, ok, failed, wall_ms, min, p50, mean, p95, p99,
max, mean_cpu_ms, mean_wait_ms`.

## Notes

- Linux only. It uses `fork()`, an anonymous shared mapping, a process-shared
  `pthread_barrier`, `getrusage()`, and `clock_gettime()`.
- Run it with the same environment a real pod uses (for example
  `CUDA_VISIBLE_DEVICES` and any memory limit variables), so the initialization
  path matches production.
- This is a measurement tool. It does not modify or optimize the lock.

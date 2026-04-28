# CUDA hook robustness audit — 2026-04-28

Reference fix: commit `03f99d7 fix(cuda): avoid NULL deref in cuMemGetInfo_v2 when caller (OptiX) crashes`

Pattern (verified against current `cuMemGetInfo_v2` body at `src/cuda/memory.c:501`):

1. Forward to the real driver first (errors surface exactly as without HAMi).
2. Early return on NULL/invalid args (driver already rejected — never deref).
3. Then HAMi enforcement / accounting logic.

Why we need this: NVIDIA Isaac Sim Kit (Carbonite / OptiX / Aftermath) calls
several of these CUDA hooks via internal probes that may pass NULL output
pointers or run before any CUDA context is current. With the current code the
HAMi enforcement path runs unconditionally, dereferences the NULL output, or
calls `cuCtxGetDevice` without checking its return — and `libvgpu.so`
SegFaults inside the app.

## Hooks needing the same pattern

Line numbers verified against current source on branch `vulkan-layer`
(`grep -n '^CUresult ' src/cuda/{memory,context}.c` 2026-04-28):

- `cuMemAlloc_v2` — `src/cuda/memory.c:135` (Task 2)
  - Body: `ENSURE_RUNNING(); allocate_raw(dptr, bytesize)` — `allocate_raw`
    will write `*dptr` and call into `oom_check` / `add_chunk` without any
    NULL guard on `dptr`.
- `cuMemAllocHost_v2` — `src/cuda/memory.c:145` (Task 3)
  - Body: forwards first (good), but on success does `*hptr = NULL` /
    re-frees via `*hptr` inside `check_oom` branch without confirming
    caller passed a non-NULL `hptr`. OOM cleanup path will crash.
- `cuMemAllocManaged` — `src/cuda/memory.c:159` (Task 3)
  - Body: `cuCtxGetDevice(&dev)` via `CHECK_DRV_API` — if no current
    context this aborts; `oom_check` runs before the real driver call so
    NULL `dptr` isn't surfaced as the driver's own
    `CUDA_ERROR_INVALID_VALUE`.
- `cuMemAllocPitch_v2` — `src/cuda/memory.c:174` (Task 4)
  - Same pattern as Managed: pre-call `cuCtxGetDevice` + `oom_check` then
    forward. NULL `dptr`/`pPitch` aren't returned by HAMi the way the
    real driver does, and post-success path writes `*dptr` into
    `add_chunk_only`.
- `cuMemHostAlloc` — `src/cuda/memory.c:223` (Task 5)
  - Forwards first (good), but OOM cleanup writes `*hptr = NULL` without
    NULL guard on the caller's `hptr` parameter.
- `cuMemHostRegister_v2` — `src/cuda/memory.c:239` (Task 6)
  - Calls `cuCtxGetDevice(&dev)` *unconditionally* and ignores its
    return code (the device variable is then unused — vestigial code).
    Also runs `check_oom` after success path. Needs forward-first +
    drop-the-stray-`cuCtxGetDevice` cleanup.
- `cuCtxGetDevice` — `src/cuda/context.c:42` (Task 7)
  - Pure passthrough today, but the underlying driver call returns
    `CUDA_ERROR_INVALID_CONTEXT` when no context is current; several of
    the hooks above currently rely on `cuCtxGetDevice` succeeding. We
    add an explicit NULL guard on `device` so HAMi's own callers
    (which may be called before any real `cuCtxGetCurrent`) don't crash
    when `device == NULL` is passed during early-init probing.

## Already robust (skip — reference patterns)

- `cuMemFree_v2` — `src/cuda/memory.c:192` (commit `3bebc8a`,
  "fix(cuda): fall back to real driver on untracked cuMemFree[Async]
  pointer"): NULL-pointer early return + fall-through to real driver on
  unknown pointer.
- `cuMemFreeAsync` — `src/cuda/memory.c:655` (same commit `3bebc8a`).
- `cuMemGetInfo_v2` — `src/cuda/memory.c:501` (commit `03f99d7`):
  forward-first + NULL guard + benign return when no current context.
  This is the canonical reference for Tasks 2–7.
- `cuMemCreate` — `src/cuda/memory.c:608` (commit `833c62c`,
  "fix: segfault in cuMemCreate hook when cuCtxGetDevice fails"):
  guards `cuCtxGetDevice` failure before HAMi enforcement.

## Out-of-scope but noted

- `cuMemFreeHost` / `cuMemHostUnregister` (`memory.c:213`, `:265`) are
  pure passthroughs and don't need hardening (they only forward to the
  real driver).
- `cuMemoryAllocate` (`memory.c:129`) is an internal helper that
  delegates to `cuMemAlloc_v2` — fixing the latter covers it.

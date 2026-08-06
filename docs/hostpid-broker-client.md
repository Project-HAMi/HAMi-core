# Host PID broker client

Status: local draft. The feature is disabled by default.

## Purpose

The client lets HAMi-core learn its host PID without creating a temporary CUDA primary context during `postInit()`. It uses the version 1 Unix socket protocol documented by HAMi and preserves the existing NVML discovery path as its fallback. A successful broker lookup still initializes NVML before utilization and context accounting use it.

## Enablement

The client is enabled only when `LIBVGPU_HOSTPID_BROKER` is the exact string `1`. Every other value leaves the current behavior unchanged.

The expected container socket is fixed at `/tmp/vgpulock/hostpid/broker.sock`. A caller cannot redirect the production client to another path.

## Trust checks

Before using a reply, the client verifies all of the following:

1. The socket path is the fixed production path.

2. The immediate directory is a real directory owned by root and is not writable by group or other users.

3. The socket is a real Unix socket owned by root.

4. The directory is visible through a read only mount.

5. The connected server peer has UID 0 according to `SO_PEERCRED`.

6. The reply has the expected magic, version, success status, and a positive PID that fits in `pid_t`.

One monotonic deadline covers connect, request write, and response read. A trickle response cannot renew the deadline.

## Fallback

If the gate is disabled or any trust, connection, deadline, or protocol check fails, `postInit()` acquires the existing process death safe lock and runs the current NVML discovery path. The broker failure does not supply a PID and does not bypass the fallback.

## CUDA context accounting

The old discovery path obtains both the host PID and the primary context size from a temporary context. The broker path skips that retain and release probe, then measures memory when the application first retains the real primary context.

After NVML starts, the broker path maps each visible CUDA device to its physical NVML index by PCI identity. This mapping does not parse `CUDA_VISIBLE_DEVICES`, so an index list and a GPU UUID use the same driver supplied identity. Devices outside the CUDA visibility set remain unmapped.

Per device limits, monitored memory, and utilization remain indexed by the visible CUDA ordinal because they describe the workload's device view. Allocation and primary context usage are indexed by physical NVML device so processes with different visibility orders still aggregate against the same GPU. Reverse lookup scans only visible CUDA devices, and an invalid or missing mapping fails closed before shared memory access.

Accounting state is kept per visible CUDA device. A nested retain adds one context charge, and the final successful release removes it. A failed add can be retried by a later retain. A failed removal stays recorded so a later retain and final release cycle can retry without adding the same charge twice. Fork handlers clear inherited accounting state and the cached parent slot before the child repeats initialization.

Hardware validation must prove that the context appears in NVML within the bounded measurement window and that the final shared region charge matches the observed primary context memory. The matrix must include a nonzero physical GPU selected by UUID and must verify the CUDA ordinal, NVML index, and primary context accounting index.

## Compatibility

1. New HAMi-core with no server uses the existing fallback.

2. Old HAMi-core ignores the new environment value and mount.

3. A rejected or unknown protocol version uses the existing fallback.

4. Existing workloads do not receive the broker mount until they are recreated.

The separate post init lock migration from PR 248 has a different mixed binary constraint and is not changed by this client.

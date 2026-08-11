# Host PID broker client

The feature is disabled by default.

## Purpose

The client lets HAMi-core learn its host PID without creating a temporary CUDA primary context during `postInit()`. It uses the version 1 Unix socket protocol documented by HAMi and keeps the NVML discovery path as its fallback. A successful broker lookup still initializes NVML before utilization and context accounting use it.

## Enablement

The client is enabled only when `LIBVGPU_HOSTPID_BROKER` is the exact string `1`. Every other value leaves the current behavior unchanged.

The expected container socket is fixed at `/tmp/vgpulock/hostpid/broker.sock`. A caller cannot redirect the production client to another path.

The companion HAMi device plugin must mount the root-owned broker directory read-only at `/tmp/vgpulock/hostpid`. The parent `/tmp/vgpulock` must be root owned and use sticky permissions if it remains writable. If this ownership, permission, or mount contract is not met, the broker lookup and node-wide fallback lock both fail safely.

## Trust checks

Before using a reply, the client verifies all of the following:

1. The socket path is the fixed production path.

2. The immediate directory is a real directory owned by root and is not writable by group or other users.

3. The socket is a real Unix socket owned by root.

4. The directory is visible through a read-only mount.

5. The connected server peer has UID 0 according to `SO_PEERCRED`.

6. The reply has the expected magic, version, success status, and a positive PID that fits in `pid_t`.

The fallback lock opens each directory component with `O_NOFOLLOW`. Every component must be root owned and use an accepted filesystem type. A group-writable or world-writable component must also have its sticky bit set. These checks reject a symlink ancestor, an ordinary writable ancestor, or an unsupported filesystem before the node lock is used. The production lock accepts ext4, XFS, tmpfs, Btrfs, F2FS, OverlayFS, ramfs, and ZFS when the build headers identify them. Other filesystem types fail with `EOPNOTSUPP` until their behavior is validated.

One monotonic deadline covers connect, request write, and response read. A trickle response cannot renew the deadline.

## Fallback

If the gate is disabled, `postInit()` keeps the existing cache-local NVML fallback. If the gate is enabled and any trust, connection, deadline, or protocol check fails, `postInit()` locks the trusted `/tmp/vgpulock/hostpid` directory before the cache record lock and runs the NVML discovery path. The node-wide lock coordinates independent cache files. A missing or untrusted broker mount causes a clear discovery failure instead of returning to cache-local discovery.

`postInit()` selects the broker result, cache-local fallback, or node-wide fallback through one tested decision function. A successful broker result selects the broker path whether the gate state is enabled or disabled, so neither fallback lock is selected. A failed broker result selects the cache-local path only when the gate is disabled and the node-wide path when it is enabled.

The enabled fallback creates one 30 second monotonic deadline before it tries the node lock. The node lock and the following cache record lock consume that same absolute deadline, so their combined acquisition cannot become two consecutive 30 second waits. This preserves the global then cache order while preventing a live holder or a reverse-order peer from hanging initialization indefinitely. A timeout produces the same clear discovery failure and releases any node lock already acquired.

The lock rechecks the opened directory after a waiter acquires `flock`. It reads current descriptor and path metadata instead of reusing the prewait mode bits. A directory that becomes writable and nonsticky while the waiter is blocked fails with `EACCES`, even when its inode, owner, mount flags, and filesystem stay unchanged.

The final check walks the path again with `O_NOFOLLOW` on every component. It compares the fresh final descriptor with the locked object and checks the current path's read only mount and filesystem policy. Replacing an ancestor with a symlink to the same final directory therefore fails even though the final device and inode are unchanged. Replacing the final object also fails after the waiter has completed its initial validation.

## CUDA context accounting

The old discovery path obtains both the host PID and the primary context size from a temporary context. The broker path skips that retain and release probe, then measures memory when the application first retains the real primary context.

After NVML starts, the broker path maps each visible CUDA device to its physical NVML index by PCI identity. This mapping does not parse `CUDA_VISIBLE_DEVICES`, so an index list and a GPU UUID use the same driver-supplied identity. Devices outside the CUDA visibility set remain unmapped.

Per device limits, monitored memory, and utilization remain indexed by the visible CUDA ordinal because they describe the workload's device view. Allocation and primary context usage are indexed by physical NVML device so processes with different visibility orders still aggregate against the same GPU. Reverse lookup scans only visible CUDA devices, and an invalid or missing mapping fails closed before shared memory access.

Accounting state is kept per visible CUDA device. A nested retain adds one context charge, and the final successful release removes it. A failed add can be retried by a later retain. A failed removal stays recorded so a later retain and final release cycle can retry without adding the same charge twice. Fork handlers clear inherited accounting state and the cached parent slot before the child repeats initialization.

Hardware validation must prove that the context appears in NVML within the bounded measurement window and that the final shared region charge matches the observed primary context memory. The matrix must include a nonzero physical GPU selected by UUID and must verify the CUDA ordinal, NVML index, and primary context accounting index.

## Compatibility

1. New HAMi-core with the broker disabled keeps the existing cache-local NVML fallback.

2. New HAMi-core with the trusted mount but no running server uses the node-wide NVML fallback.

3. New HAMi-core with the broker enabled but without the trusted mount fails host PID discovery safely.

4. Old HAMi-core ignores the new environment value and mount.

5. A rejected or unknown protocol version uses the node-wide fallback when the trusted mount exists.

6. Existing workloads do not receive the broker mount until they are recreated.

The separate post-init lock migration from [PR 248](https://github.com/Project-HAMi/HAMi-core/pull/248) has a different mixed-binary constraint and is not changed by this client.

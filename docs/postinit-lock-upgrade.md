# Post init lock upgrade contract

Status: required for upgrades across the semaphore to record lock boundary.

## Why a rolling upgrade is unsafe

Older HAMi-core processes serialize host PID discovery with
`sem_postinit`, a process shared unnamed semaphore inside the shared cache.
Newer processes keep that field for layout compatibility but serialize with a
POSIX record lock at a fixed offset in the cache file.

The two primitives do not observe each other. An old process can hold the
semaphore while a new process holds the record lock, so both can execute host
PID discovery at the same time. The cache version fields do not prevent this.
Existing binaries only report a version mismatch and continue.

The trusted broker path does not create a temporary CUDA context and does not
take the post init lock. After validating the broker response, it initializes
NVML, maps visible CUDA devices by PCI identity, and updates the process slot
under the shared region lock. It can overlap with a locked NVML fallback because
it does not enter the context probe section. A rejected broker response uses the
fallback protected by the record lock described above.

The local mixed protocol probe reproduces the overlap directly. It is under
`evidence/issue-1662/lock-benchmark/mixed_protocol_probe.c` in the companion
evidence tree.

## Required upgrade

1. Cordon the node and stop every GPU workload that can map the affected
   shared cache.

2. Confirm that no HAMi-core process still maps any affected cache. Check active
   memory mappings for every effective `CUDA_DEVICE_MEMORY_SHARED_CACHE` path,
   including generated per-container `usage.cache` files and the
   `/tmp/cudevshr.cache` fallback. An open descriptor check is not sufficient.

3. Remove each cache only after every process retaining a mapping has exited.

4. Install one HAMi-core revision across the node.

5. Restart the device plugin and recreate workloads.

6. Validate one low-density workload before uncordoning the node. Confirm host
   PID detection, memory accounting, and record lock acquisition.

Do not delete or replace a cache while a process still maps it. Unlinking the
path does not move an existing mapping to the new file and can split accounting
between two cache objects.

## Required rollback

Rollback across this boundary uses the same drain procedure. Stopping only the
device plugin is insufficient because workload processes contain HAMi-core and
retain their cache mappings.

## Deployment guarantee

The current implementation does not support a mixed-version grace period and
does not fail closed when an old process joins. Cluster rollout policy must
prevent mixed HAMi-core revisions on a shared cache. A future rolling protocol
would need an enforced version handshake that both old and new binaries obey.

---
name: Bug Report
about: Report a problem encountered while using HAMi-core
labels: bug
---

<!-- Please use this template while reporting a bug and provide as much info as possible. Not doing so may result in your bug not being addressed in a timely manner. Thanks!
-->

**What happened**:

**What you expected to happen**:

**How to reproduce it (as minimally and precisely as possible)**:

**Anything else we need to know?**:

- The output of `nvidia-smi -a` on your host
- Your Docker or containerd configuration file
- The HAMi-core logs and relevant `LIBCUDA_LOG_LEVEL`
- The build and workload reproduction commands
- Relevant `CUDA_DEVICE_MEMORY_LIMIT`, `CUDA_DEVICE_SM_LIMIT`, and `LD_PRELOAD` values
- Any relevant kernel output lines from `dmesg`

Before posting, remove or mask credentials, tokens, and other sensitive data from configuration and logs.

**Environment**:
- HAMi-core version, commit, or image:
- GPU model:
- NVIDIA driver version:
- CUDA toolkit/runtime version:
- Docker or containerd version:
- Build/run command, image, and tag used:
- Workload or framework:
- Kernel version from `uname -a`:
- Others:

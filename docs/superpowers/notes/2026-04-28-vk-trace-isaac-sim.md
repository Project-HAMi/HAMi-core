# Vulkan layer trace — Isaac Sim Kit init under LD_PRELOAD (2026-04-28)

Step C Task 3 trace. **Conclusion: Step C Tasks 1+2 introduce a regression on the LD_PRELOAD-only (no implicit-layer manifest) code path.** Production .so has been restored to the pre-Step-C build pending plan revision.

Build base under test: HAMi-core `vulkan-layer` after Step C Tasks 1+2 (`eea2beb`).
Build under comparison: HAMi-core `vulkan-layer` Step B end (`7dcb5a4`) — md5 `8f889313ece246b2d08ea6291f48b67a` on `/usr/local/vgpu/libvgpu.so.bak-pre-step-c`.
New build md5: `9586feee3f0672ab35c4d6f63120dfc4`.

## Methodology

ws-node074 isaac-launchable namespace, pod `isaac-launchable-0-757b765f45-2d76x` (vscode container).
50s `runheadless.sh` runs with `ACCEPT_EULA=y`. `pkill -KILL kit` between runs.

## Results

### Test matrix (3 isolated runs)

| Run | LD_PRELOAD | implicit layer manifest | exit | crash | listen :49100 | HAMi-core init |
|---|---|---|---|---|---|---|
| 1. Baseline | none | none | 124 | 0 | 1 | n/a |
| 2. Pre-Step-C (`7dcb5a4` backup) | `.so.bak-pre-step-c` | none | 124 | 0 | (alive) | 0 (timeout reached) |
| 3. Post-Step-C (`eea2beb`) | new `.so` | none | **139** | **2** | (n/a; crashed at 1.5s) | 9 |
| 4. Post-Step-C + `/tmp/vk-layers/hami.json` (process-scope) | new `.so` | `VK_LAYER_PATH=/tmp/vk-layers VK_INSTANCE_LAYERS=VK_LAYER_HAMI_vgpu` | **139** | **1** | (n/a) | 8 |

Run 4 attempted to enable our layer via process-scope `VK_LAYER_PATH` because writing into `/etc/vulkan/implicit_layer.d/` was sandbox-blocked. **`HAMI_VK_TRACE=1` produced 0 trace lines in Run 4** — the loader did not actually invoke our `vkGetInstanceProcAddr`, suggesting `VK_INSTANCE_LAYERS` is the explicit-layer mechanism and does not apply to NVIDIA's implicit-layer-shaped manifest. We could not get trace evidence under a manifest-activated path in this session.

### HAMI_VK_TRACE lookup names (Run 3 + Run 4)

**0 lines in both.** Vulkan layer init never reached our wrappers — Kit crashed before vkCreateInstance.

### Crash backtrace (Run 3)

```
000: libc.so.6!__sigaction+0x50
001: libEGL_nvidia.so.0!__egl_Main+0x3b3
002: libEGL_nvidia.so.0!__egl_Main+0x1a27
003: libGLX_nvidia.so.0!vk_icdNegotiateLoaderICDInterfaceVersion+0x3b9
004: libGLX_nvidia.so.0!__glx_Main+0x2b2d
005: libGLX_nvidia.so.0!vk_icdNegotiateLoaderICDInterfaceVersion+0x12
006: libvulkan.so.1!+0x31724
007: libvulkan.so.1!+0x32033
008: libvulkan.so.1!+0x31db0
```

NVIDIA driver 580.142, RTX 6000 Ada Generation, 46068 MiB.

This is the Vulkan loader (`libvulkan.so.1`) loading the NVIDIA ICD (`libGLX_nvidia.so.0`'s `vk_icdNegotiateLoaderICDInterfaceVersion`), which dispatches into NVIDIA's EGL backend (`libEGL_nvidia.so.0!__egl_Main`), which crashes inside `__sigaction` setup. The crash is during ICD initialization — before any `vkCreateInstance` could have run, so our layer's `g_first_next_gipa` was still NULL.

## Hypothesis (root cause)

LD_PRELOAD-only path (no implicit-layer manifest):

1. Our `libvgpu.so` exports `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr` with default visibility — these symbols enter the global `RTLD_DEFAULT` namespace at process start.
2. NVIDIA's ICD (`libGLX_nvidia.so.0`) initializes through `libvulkan.so.1` and resolves several Vulkan entry points by name. Some of those resolutions reach our exported symbols instead of the loader-routed ICD entry, because we are LD_PRELOAD'd ahead.
3. Step B's `7dcb5a4` exported the same wrappers but with a NULL-on-unknown-instance fallback for `GIPA`/`GDPA`, so those paths returned NULL → ICD treated as "function not present" → ICD self-resolved or short-circuited gracefully.
4. Step C Tasks 1+2 changed two things on top of that:
   - **Task 1:** added `HAMI_HOOK(EnumerateDeviceExtensionProperties)` and `HAMI_HOOK(EnumerateDeviceLayerProperties)` — meaning a global GIPA lookup for those names now returns our wrapper instead of falling through.
   - **Task 2:** changed unknown-instance fallback in `hami_vkGetInstanceProcAddr` / `hami_vkGetDeviceProcAddr` from "return NULL" to "forward via cached `g_first_next_gipa`".
5. Under LD_PRELOAD-only, our `g_first_next_gipa` is NULL during ICD bring-up because `vkCreateInstance` has not yet run. So Task 2's fallback collapses back to the original NULL return — it should be no worse.
6. The remaining suspect is Task 1: the new `EnumerateDeviceExtensionProperties` wrapper, when `g_inst_head == NULL`, returns `pPropertyCount=0, VK_SUCCESS`. NVIDIA's ICD/EGL backend may consult device extensions during EGL bringup and treat 0 entries as a hard error, or NULL-deref a result pointer it expected to be populated. Pre-Step-C had no hook for that name, so the GIPA chain returned NULL → NVIDIA fell back to its own internal table.

This is a hypothesis from comparative evidence + backtrace, not from a working trace. The smoking-gun trace would be a HAMI_VK_TRACE log under a properly-activated implicit-layer manifest, which the current sandbox prevents.

## Decision (Task 4)

**Stop and surface to controller.** The Plan's Task 4 was framed as "add hooks for additional vkGetPhysicalDevice* names that returned NULL in trace" — an additive change. The actual evidence calls for the opposite: **revisit Tasks 1+2 because they introduced a runtime regression on the LD_PRELOAD-only path, which is the path Plan Task 6 explicitly verifies.**

Options for the controller:

1. **Revert Task 1's EnumerateDevice* hooks for the unknown-instance case** — return NULL (pre-Step-C behavior) when `g_inst_head == NULL`, and only forward to `hami_instance_first()` when an instance is registered. Preserves the Carbonite-fix intent for the manifest path while staying inert on LD_PRELOAD-only.
2. **Gate the layer's wrapper exports on a HAMI-mode check** — at .so init, detect whether our layer manifest is active (e.g., via `VK_LAYER_HAMI_vgpu` in `VK_INSTANCE_LAYERS` or environment) and turn the GIPA/GDPA + Enumerate hooks into pass-throughs otherwise.
3. **Dlsym(RTLD_NEXT) fallback** — when `g_first_next_gipa == NULL`, resolve `vkGetInstanceProcAddr` via `RTLD_NEXT` and forward, so ICD init via LD_PRELOAD never sees our hooks return surprising values.

Option 1 is the smallest delta and most likely to keep the Step B regression-pass intact.

## Update (Run 5): hypothesis falsified

A Task 1-targeted fix was attempted: `HAMI_HOOK_GATED(EnumerateDeviceExtensionProperties)` and `HAMI_HOOK_GATED(EnumerateDeviceLayerProperties)` — return our wrapper only when `hami_instance_first() != NULL`, fall through to NULL otherwise.

| Run | Build | exit | crash | listen | trace lines |
|---|---|---|---|---|---|
| 5. Step C + gated hooks (md5 1048daaf) | new | **139** | **2** | 0 | **0** |

Same crash, same backtrace, same `HAMI_VK_TRACE=0 lines`. The gate addressed the only theory we had for how Tasks 1+2 could affect ICD init, and it changed nothing.

**Conclusion:** the regression is NOT triggered by NVIDIA ICD calling our `vkGetInstanceProcAddr`. Our wrapper is genuinely never called — `HAMi-core init=9` confirms `LD_PRELOAD` succeeded but the Vulkan loader code path that would call our GIPA hasn't run by the time of the crash.

Yet the new build crashes and the old build (`8f889313`) does not. So the differential is somewhere that runs **at .so load time or earlier in NVIDIA driver init**, not in our Vulkan wrappers. Candidate diff surface:

- ELF symbol exports added by Step C (e.g., `hami_instance_first` is now a non-static external) — could collide with a NVIDIA driver weak symbol or change global lookup order.
- Static initializer / constructor side effects from new TUs being linked in (`dispatch.c` now compiles slightly more code).
- Any change between `8f889313`'s source state and `7dcb5a4` that we have not yet identified — the backup md5 was created before this session and may NOT correspond to commit `7dcb5a4`. We cannot bisect further without rebuilding `7dcb5a4` cleanly on ws-node074, which the next attempt was sandbox-blocked.

## Diagnostics not yet possible in this session

- Build commit `7dcb5a4` cleanly and md5-compare to `8f889313` — would confirm whether prod backup is from Step B end or some earlier commit.
- `nm -D libvgpu.so` symbol diff between `8f889313` and `9586feee` — would reveal new exports.
- `LD_DEBUG=symbols,bindings` under runheadless — would show which symbol resolution actually picks up our `.so` first.

## Cleanup performed (this session)

- Restored `/usr/local/vgpu/libvgpu.so` from `.bak-pre-step-c` on ws-node074. md5 `8f889313ece246b2d08ea6291f48b67a` confirmed (3x: after Run 3, after Run 4, after Run 5).
- Removed pod `/tmp/vk-layers` and `/tmp/vk-trace`.
- Confirmed isaac-launchable-0 baseline still alive after each restore: `exit=124 crash=0 listen=1`.
- Discarded the unproven gate edit from `src/vulkan/layer.c`.

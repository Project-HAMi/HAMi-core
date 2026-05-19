# Vulkan dispatch lifetime + chain copy audit (2026-04-28)

Step C Task 5 audit. Read-only review of `src/vulkan/dispatch.c` and `src/vulkan/layer.c` against Vulkan loader spec 1.3 §38.

## Scope

1. Lifetime of `hami_instance_dispatch_t` / `hami_device_dispatch_t` returned by `hami_instance_lookup` / `hami_device_lookup` / `hami_instance_first`.
2. In-place advance of `chain->u.pLayerInfo` in `hami_vkCreateInstance` / `hami_vkCreateDevice`.

---

## 1. Dispatch lifetime

### Code reviewed

`hami_instance_lookup` (`dispatch.c:49-55`):

```c
hami_instance_dispatch_t *hami_instance_lookup(VkInstance inst) {
    pthread_mutex_lock(&g_lock);
    hami_instance_dispatch_t *p = g_inst_head;
    while (p && p->handle != inst) p = p->next;
    pthread_mutex_unlock(&g_lock);   // <-- lock dropped here
    return p;                         // <-- caller uses p outside lock
}
```

Same pattern in `hami_device_lookup` (`dispatch.c:96-102`) and `hami_instance_first` (`dispatch.c:42-47`).

`hami_vkDestroyInstance` (`layer.c:101-106`):

```c
hami_instance_dispatch_t *d = hami_instance_lookup(instance);
if (d) d->DestroyInstance(instance, pAllocator);
hami_instance_unregister(instance);   // frees the node
```

### Race analysis

**Theoretical race:** Thread A calls `hami_vkDestroyInstance(I)`. Thread B simultaneously calls `vk*` on `I`. Both lookups succeed; Thread A then unregisters and frees the node. Thread B then dereferences a freed dispatch pointer → use-after-free.

**Spec position (Vulkan 1.3 §3.6 "Threading Behavior"):**

> Externally synchronized parameters: The application MUST ensure that no two
> calls operate on the same handle simultaneously when at least one of them
> is `vkDestroy*`.

VkInstance, VkDevice, VkQueue, VkCommandBuffer (and command pool) are externally synchronized. The application — not the layer — is responsible for serializing destroy against any concurrent use.

**Real callers:**
- NVIDIA Carbonite (`libcarb.graphics-vulkan`) destroys VkInstance / VkDevice on a single shutdown thread after stopping all rendering.
- Isaac Sim Kit follows the same pattern.

**Use of `hami_instance_first()`:**
Called only from the layer's `vkEnumerateDevice*` hooks (`layer.c:263`, `layer.c:279`), which run during normal lifecycle (after CreateInstance, before DestroyInstance). It does not race with destroy under the spec's external-sync requirement.

### Decision: no code change

The lookup-then-use pattern relies on spec-mandated external synchronization that real-world Vulkan applications satisfy. Adding refcounts or extending the lock across the call would be a behavioral divergence from how every reference layer (Khronos `VK_LAYER_KHRONOS_validation`, `nvidia_layers.json`) handles it — those also drop the lock before invoking the next-chain function pointer, for the same reason: holding a global mutex across an unbounded next-chain call would deadlock.

Documented but not patched. If we ever observe a real crash whose stack matches the use-after-free pattern (Thread B inside a hooked entry point with stale dispatch), revisit with refcounts.

---

## 2. Chain pLayerInfo in-place advance

### Code reviewed

`hami_vkCreateInstance` (`layer.c:75-76`):

```c
PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
```

Same pattern in `hami_vkCreateDevice` (`layer.c:117-118`).

### Spec / reference layers

**Vulkan-Loader-Interface (`docs/LoaderLayerInterface.md`, Khronos):**

> Layers should follow the same recommendation as drivers and advance the
> link information before calling down. ... The loader requires that the
> layer advance the link node so that subsequent layers see the correct
> next-link.

This is the canonical instance-chain pattern documented in the Khronos vulkan-loader source (`loader/loader.c` and the per-layer examples in `tests/framework/layer/`).

**Reference layer implementations all do in-place advance:**

- `VK_LAYER_KHRONOS_validation` (`layers/state_tracker/instance_state.cpp`):
  ```cpp
  chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;
  ```
- NVIDIA's optimus / nvoptix layers (per Khronos sample layers and public layer skeletons): same pattern.
- Renderdoc's `VK_LAYER_RENDERDOC_Capture`: same pattern.

The in-place advance is the documented recommendation, not a workaround.

### Reuse concern

**Question:** Does NVIDIA driver or Carbonite reuse the same `VkInstanceCreateInfo` after our advance, causing the chain to skip a layer?

**Investigation:** `pCreateInfo` is `const VkInstanceCreateInfo *`; the Vulkan loader allocates a fresh `VkLayerInstanceCreateInfo` per layer per `vkCreateInstance` invocation (see Khronos `vulkan-loader` `loader/trampoline.c::terminator_CreateInstance`). The structure is loader-owned scratch memory, not application memory, so reusing it across calls is not possible — the loader builds a new chain on every call.

**Verified:** within a single `vkCreateInstance` invocation, each layer in the chain advances the same `pLayerInfo` once before calling down, by design. After our advance, the next layer (or driver) sees its own link as the head, not ours. This is exactly the spec contract.

### Decision: no code change

Pattern matches spec recommendation and every reference layer. A deep-copy would diverge from the canonical pattern and add allocation overhead with no behavioral gain.

---

## Conclusion

Both audit areas are clean. Step C does not need additional code patches from this audit. If runtime evidence later contradicts the analysis, the items to look for are:

- **Lifetime:** stack with hooked entry-point on Thread B, freed dispatch on Thread A. Fix path: refcount the dispatch struct or extend the lock with a try-lock + retry.
- **Chain:** vkCreateDevice receiving a `pLayerInfo` that already points past the next layer. Fix path: deep-copy the `VkLayerDeviceCreateInfo` and patch the copy, restore on return.

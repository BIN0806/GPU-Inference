# ADR 0009 — Standalone unfused CUDA control before bindings and fusion

Date: 2026-10-26. Status: implemented; host descriptors tested; native compilation
and hardware checks pending at the initial commit.

Start with separate residual-add and RMSNorm launches, both using caller-owned
buffers on an explicit stream. Store and reload the rounded residual so the
contract is visible. Use a 128-thread shared-memory tree as the first reduction.
Keep CUDA optional and the allocator independent of both CUDA and PyTorch.

Alternatives considered:

- Starting with fusion hides the intermediate store and removes the native
  separate-kernel control needed to distinguish launch savings from other changes.
- Building a PyTorch extension immediately provides convenient tensors but adds
  framework/compiler ABI and ownership concerns before validating native kernels.
  A standalone library/checker makes this increment independently compilable.
- Warp reductions are likely useful later, but add lane-mask reasoning before
  establishing a shared-memory reference. No performance preference is claimed.
- A raw unchecked pointer API is small but defers too many shape/alias bugs to
  the GPU. CPU descriptor checks plus device/stream identity checks reject those
  failures first. Truthful capacities and valid bounded values remain preconditions.
- Silently synchronizing inside the wrapper simplifies ownership but obscures
  stream behavior and benchmarking. The caller owns lifetime through completion;
  errors can leave earlier launches pending and are not atomic rollback.

Native rows are capped at INT32_MAX (one block per row), width at 8192. Empty
batches are shape/type-checked no-ops with empty buffers. Managed memory, strides,
BF16, in-place outputs, cross-device pointers, and gradients are outside scope.
The public tensor reference remains unchanged and has its own stricter boundary
checks even for an empty batch.

Use a standard CPU CI runner to compile/link sm_86 and sm_89 and run host-only
checks against a pinned minimal NVIDIA toolchain. This costs no GPU rental and
is explicitly weaker than GPU execution. Full numerical, memory, race, stream,
and performance validation remains a hardware gate. Revisit the binding boundary,
launch checks, and reduction only after that evidence exists.

See [native contract and commands](../CUDA_RMSNORM.md), [numerical contract](../RMSNORM.md),
and engineering entry 0013.

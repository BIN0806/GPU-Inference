---
name: kvlab-cuda-kernels
description: "Design, review, or debug this project's CUDA residual/RMSNorm and paged decode kernels, focusing on numerical contracts, indexing, reductions, and evidence-driven optimization."
---

# CUDA kernel engineering

Read [project guidance](../../../AGENTS.md), recent [log entries](../../../docs/ENGINEERING_LOG.md), and the relevant Phase 2/3 sections of [ROADMAP.md](../../../ROADMAP.md). Preserve the user's learning objective and the requested implementation/review scope.

## Establish the kernel contract

Write down shapes, strides, dtype, scale/epsilon, intermediate rounding, aliasing, supported lengths, output dtype, and stream. For residual/RMSNorm, the planned contract returns both rounded residual and normalized output; comparing that with an output-only or unrounded operation is invalid. Confirm the actual code contract before applying the recommendation.

For paging, trace a token through logical page, physical ID, in-page offset, KV head, and channel. Use wide offsets and independently shuffled page mappings. For GQA, verify Hq is divisible by Hkv and that storage/mapping follows Hkv. Enforce valid lengths before loads. An empty batch can be a host no-op; empty active contexts need explicit rejection or semantics.

For decode SDPA references, inspect non-square causal-mask alignment. A one-token query against a valid past/current cache must see all valid keys. For online softmax, verify both accumulated denominator and vector are rescaled when the running maximum changes; handle all-invalid tiles. Split-context outputs need a statistics-aware merge, not an average.

## Correctness before performance claims

Minimize to one request/head and two shuffled pages, or one odd-width normalization row. Compare with an independent high-precision oracle built from the same quantized inputs. Inspect errors by coordinate and stage, not only one maximum. Do not widen tolerance to hide structured errors.

Audit masks, participating lanes, barriers, shared-buffer reuse, alignment, scalar tails, and launch geometry. Check launch and execution errors separately. Use relevant Compute Sanitizer modes on small cases when a GPU is available and authorized. A sanitizer pass does not prove host/GPU lifetime safety.

Optimize one mechanism at a time: thread mapping, reduction, live registers versus reloads, vectorization, tile size, or layout. State the hypothesis and expected regime; record resource use and compare to a preserved baseline. Occupancy is a constraint, not the objective. Packed layout changes must account for append/packing costs and coordinate with integration owners.

Return the math/indexing explanation, concrete finding or exercise, smallest verification cases, executed evidence, and untested assumptions. Use [the experiment template](../../../docs/templates/EXPERIMENT.md) when measurement is part of the task. Do not provision resources or broaden the support matrix merely to complete a review.

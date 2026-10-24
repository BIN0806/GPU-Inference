# ADR 0001: scope, cache ownership, and safe reclamation

Date: 2026-10-23. Status: **recommended design; not yet implemented or empirically validated**.

## Context

The project must demonstrate low-level reasoning while allowing local development and remaining below $100 infrastructure cost. It needs genuine allocator/kernel integration without taking on all of model serving.

## Decision

Build a single-layer synthetic paged-decode benchmark plus an independent fused residual/RMSNorm track. The allocator manages CPU page IDs/refcounts; a GPU owner retains persistent K/V tensors. Start with P=16, `[N,Hkv,P,D]` layout, one CPU owner thread, one stream, and synchronized step boundaries. Support explicit immutable full-page sharing; defer partial-page copy-on-write and automatic prefix lookup. Add event retirement only after the serialized path passes its tests.

## Alternatives considered

| Alternative | Advantage | Cost / reason deferred |
|---|---|---|
| Per-request device allocations | Very direct ownership | Growth/copy overhead, reservation waste, and driver/framework behavior obscure the allocator study |
| Full end-to-end model first | Immediate generated text | Integration volume hides fundamental kernel and lifetime bugs |
| Atomic refcounts and multiple scheduler threads | Possible concurrency | Adds contention and ordering without solving GPU completion; no measured need yet |
| Return pages as soon as CPU refcount reaches zero | Simple and fast | Unsafe when a submitted reader still has the physical ID |
| Packed vLLM-style layout immediately | Potentially better transactions | More indexing/append complexity before a clear reference exists |
| Share arbitrary partial pages | Better granularity | Requires copy-on-write, copy ordering, rollback, and extra lifetime tests |

## Invariants

Every physical page is Free, Live, or Retired. Free pages occur once in the free list. Live references count all request/cache owners. Retired pages are unavailable until a fence covers every submitted reader. Request table and staging lifetime extend through their GPU use. Allocation/append failure leaves the previous request state intact. The arena/pool outlive requests and pending submissions.

## Consequences

CPU allocator development can proceed without a GPU. Synchronization limits overlap but makes the initial proof manageable. A persistent arena exposes logical capacity without repeated `cudaMalloc`. Physical fragmentation inside the equal-page pool is avoided; tail waste and device/framework reservations remain. Sharing saves storage but can reduce locality or add control work.

## Validation and revisit conditions

Validate rollback, randomized page mappings, sibling prefix divergence, and delayed reader/reuse. Measure table-copy and synchronization time before adding overlap. Revisit page size/layout with a fixed-byte-budget sweep. Revisit multi-threading only if CPU control is a demonstrated bottleneck. Revisit full-model integration only after the core report and budget gates are complete.

No performance result is asserted by this decision record.

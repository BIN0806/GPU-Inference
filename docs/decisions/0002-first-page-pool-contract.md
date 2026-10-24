# ADR 0002: the first page-pool implementation contract

Date: 2026-10-23. Status: **implemented; validation linked from the first blog installment**.

## Context

The user authorized implementing the roadmap in small chunks suitable for a technical blog series. The first chunk needs an executable allocation primitive and a clear boundary from request sharing and GPU execution.

## Decision

Implement a single-threaded, noncopyable/nonmovable CPU `PagePool`. A preallocated stack holds free IDs; a per-page slot holds allocated state and a 64-bit generation. An allocation returns an optional `{id,generation}` handle, exhaustion returns nullopt, and invalid releases return a typed status without modifying state. A handle is a token scoped to its issuing pool, not a self-releasing owner and not proof of cross-pool provenance.

Capacity is bounded by INT32_MAX for the planned signed device page-table representation. Construction can still fail for insufficient host memory. Generation overflow throws before mutation and is a terminal condition for this pool instance; it is not permitted to wrap. Ordinary valid allocation/release paths do no dynamic allocation after construction.

`validate_invariants()` is an O(capacity) diagnostic using temporary storage. It is excluded from any future hot-path benchmark. GoogleTest v1.17.0 is a test-only dependency pinned by archive SHA-256; a tests-off build requires no GoogleTest download.

## Alternatives and consequences

- Bare IDs plus a live bit are simpler, but a stale ID could release a newer allocation. Generations address that failure within one pool lifetime.
- Embedding pool provenance or using a process-global handle namespace would detect more cross-pool misuse but adds scope and metadata. This first API requires the caller to use the issuing pool.
- Immediate reference counting would enable sharing but broaden the first post. It follows in part 02; this version has one logical owner per allocation.
- A tree/set or variable-sized allocator has different space/work tradeoffs. A stack gives constant-work transitions for equal-sized pages and a simple uniqueness invariant.
- An RAII page lease is deferred until request ownership is introduced; copying the current plain token does not add ownership or release automatically.

## Validation and limits

The focused tests cover capacity, uniqueness, exhaustion, reuse, invalid/double/stale releases, and arbitrary release order. A separate set-based model drives 16 × 10,000 seeded operation steps. See part 01 evidence (local-only record) for actually executed configurations/results. No latency claim, GPU safety guarantee, request rollback, prefix sharing, or concurrency support is established.

Revisit the contract when adding reference counts/request RAII, multiple interacting pools, or asynchronous readers. Preserve stale-handle tests as the ownership system grows. GPU reclamation will require completion fencing even when a CPU generation check succeeds.

## Amendment — chunk 02

The one-owner implementation above is historical. [ADR 0003](0003-sealed-pages-and-leases.md) replaces its live flag with checked references and adds sealing and ownership leases. `Released` now means one reference dropped. It does not necessarily return a physical page to the free list. Chunk 01 evidence and the source transition remain linked from the blog series.

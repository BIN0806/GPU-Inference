# ADR 0007 — Reuse request-table storage and roll back the acquired suffix

Date: 2026-10-25. Status: implemented, measured, and tested locally and on Linux. Supersedes only the growth transaction in
[ADR 0004](0004-request-transactions.md). Sharing and move assignment retain
their original candidate-table transactions.

## Trigger and alternatives

Reservation previously allocated a complete candidate table on every new page
boundary, initialized it, acquired the new pages, moved existing owners, then
swapped. That gives a clear commit point, but page-at-a-time growth performs
quadratic total metadata work. Chunk 04 measured frequent table allocations.

- Keep full candidates: simplest rollback and no retained larger table after a
  page-acquisition failure, but repeated allocation and relocation.
- Reserve a maximum-length table at request creation: predictable steady state,
  but requires a request limit and pays host memory before it is needed.
- Keep a separate acquisition journal: isolates old storage, but the journal
  itself needs storage and another move/ownership path.
- Grow the live vector geometrically, using its new suffix as the transaction:
  fewer allocation calls and no extra journal, but distinguish logical rollback
  from retained host capacity. Choose this within the single-thread contract.

## Algorithm and proof obligations

1. Validate attachment/length and preflight physical free pages. A pool-capacity
   rejection must not allocate host metadata.
2. If storage is insufficient, request `max(needed, min(2*capacity, limit))`,
   where `limit=min(vector.max_size(), pool.capacity())`. Check the bound and
   doubling before arithmetic. A vector may grant more than requested; the
   request bound is not a portable promise about its exact capacity.
3. Reserve before acquiring pages. Noexcept lease moves preserve ownership
   across relocation. A failed host allocation leaves the request unchanged.
4. Resize into already available storage with empty noexcept leases. Fill only
   entries after the old size. On failure, pop the entire provisional suffix;
   RAII releases each successfully acquired reference once.
5. Publish `reserved_length` after all acquisitions succeed. There are no
   concurrent observers under this CPU interface's contract.

Static assertions enforce the no-throw construction/move/destruction properties
used by this reasoning. Values returned by `page_handle` remain valid; the API
never exposes references to vector elements. Reallocation's pointer invalidation
is therefore internal. C++ specifies reserve's allocation/exception behavior and
insertion within reserved capacity; see the
[working draft's vector capacity contract](https://eel.is/c++draft/vector.capacity).

## Failure boundary and memory cost

Existing handles, lengths, page references, seals, and physical free capacity
are preserved on failure. Brief acquisitions can advance generations and reorder
free IDs, just as before. A completed host `reserve` can remain after later page
acquisition fails. This is deliberate retained capacity, not a page leak or a
full rollback of allocator-resource observations. Reset clears leases while
keeping storage; destruction returns it to the borrowed resource.

Repeated growth to N pages now requests O(log N) host reallocations and moves
O(N) existing entries in total. One growth call is still O(old pages + new pages)
when it reallocates, or O(new pages) with sufficient capacity. It is **not** a
worst-case constant-time operation. Prefix sharing and resource-sensitive move
assignment remain O(number of transferred entries) candidate transactions.

Retained metadata can accumulate across idle request slots. The request bound
uses total pool capacity, so a high-water mark can greatly exceed a request's
current table size. Sharing may replace that storage on the next admission.
Measure both peak and post-drain bytes; do not equate fewer allocations with
lower host memory. Add explicit quiescent trimming or an up-front limit only if
measured budgets justify the extra policy. Do not put shrinking on the hot path
without evaluating the next admission's allocation cost.

## Evidence and revisit

New tests cover no-allocation boundary growth, at most seven allocations for
64 page-by-page reservations, reset reuse, preflight ordering, and exceptions
at every new suffix position while preserving a shared prefix. Existing
allocation-failure and independent model tests remain. The experiment requires
identical event outcomes and sampled physical accounting before timing comparison.

Revisit for concurrent request writers, exposed vector views, asynchronous GPU
retirement, or a strict per-request host-memory budget. Those change assumptions;
an atomic refcount alone does not make this transaction concurrent or GPU-safe.
See [benchmark protocol](../BENCHMARKS.md) and engineering entry 0010 for results.

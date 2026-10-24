# ADR 0003: sealed pages and explicit ownership leases

Date: 2026-10-23. Status: implemented and tested in Debug, Release, and UBSan-only builds; ASan validation remains open.

## Problem and decision

A shared page must survive any one owner's completion. Replace the allocated flag with a non-atomic 32-bit reference count. Allocation starts at one; successful retain adds one; release drops one and returns physical capacity only at zero. `ReleaseResult::Released` means one reference released, not necessarily a free page. Generation changes only on a new allocation, never on retain, seal, or partial release.

Add positive `tokens_per_page` geometry (default 16) and `seal_full(handle, valid_tokens)`. A current live page may seal only when the caller-declared valid-token count equals the geometry. Once sealed, it cannot become writable again during that allocation, even when one owner remains. Retain requires a sealed page. A page with more than one reference must be sealed; a free page must have zero references and no seal. Reallocation clears the seal and starts a new generation.

The caller attests that data is complete: no KV payload or token-write tracking exists yet. Sealing is metadata permission, not hardware protection. Later request/table and write APIs must enforce this decision. Pool statistics count physical pages; per-page inspection reports logical references.

Add a move-only `PageLease` owning one reference, with factory allocation, explicit `try_share`, reset, and noexcept move/destruction. The pool outlives its leases. The lease stores pool provenance, but its borrowed handle still does not. Copies of a borrowed handle never retain. Successful retain is followed only by nonthrowing lease construction/movement. Invalid underlying release during lease destruction terminates as an ownership contract violation; it is not a normal allocation-failure path.

## Alternatives and tradeoffs

- **Bare handles with manual retain/release:** compact and useful for low-level tests, but generations identify an allocation, not distinct owners. Releasing one owner's same handle twice while another reference remains can steal the other's reference. A count alone cannot detect that misuse. Keep manual API obligations explicit and use leases for normal ownership; never manually release a lease's borrowed token.
- **Owner IDs for every retain:** can validate individual releases but needs owner records and lifetime management beyond this chunk. Revisit for untrusted clients or complex cross-subsystem transfers.
- **Implicitly copyable shared ownership:** convenient, but copying containers silently changes counts. Explicit sharing and move-only leases make those transitions reviewable. A separate `shared_ptr` control block would also add machinery when per-page metadata already exists.
- **Unseal at reference count one:** seems to recover mutability but invalidates the persistent immutable-page contract and complicates prefix reasoning. Require a new allocation for any future writable copy.
- **Allow partial-page sharing now:** requires copy-on-write, copying payloads, and transactional mapping changes. Instead, share complete pages and allocate independent tails.
- **Atomics:** unnecessary under the single-thread contract and insufficient for GPU completion. Revisit the entire synchronization model before making metadata concurrent.
- **Overflow testing by billions of retains:** expensive and unnecessary. A test-only friend injects a near-maximum count, exercises the real public boundary, then restores a valid owner count. It is defined only in tests and exposes no runtime test switch.

## Verification and revisit gates

Keep chunk 01 regressions. Add P-1/P/P+1 seal boundaries, final-release-only reclamation, count overflow, stale seal/retain, all six release orders for three owners, move assignment across pools and within a shared page, repeated reset, exception unwinding, and owner-list model checking. See chunk 02 evidence (local-only record) for executed results.

This is a metadata-only explicit sharing primitive and demo, not complete request-level `share_prefix`, automatic lookup, payload deduplication, or serving. Request growth/rollback is next. GPU work requires a completion-aware retirement stage: neither generation validity nor reference count zero is a completion fence.

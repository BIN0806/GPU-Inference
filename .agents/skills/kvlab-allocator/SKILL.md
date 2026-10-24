---
name: kvlab-allocator
description: "Design, review, or debug the LLM Kernel Inference CPU KV page allocator, request ownership, prefix sharing, and rollback. Use for allocator invariants and tests, not standalone CUDA optimization."
---

# KV allocator engineering

Read the project's [guidance](../../../AGENTS.md), recent [engineering log](../../../docs/ENGINEERING_LOG.md), and the relevant Phase 1/ownership sections of [ROADMAP.md](../../../ROADMAP.md). The user is the primary implementer: match the requested design/review/debug scope; prefer a concrete exercise or counterexample over unsolicited complete source code.

## Review the state machine before the data structure

Establish the physical-page state, every logical/cache owner, request length, page-table entries, and failure contract. Verify that each free ID occurs once, live references count all owners, and retired pages cannot be allocated until their readers complete. Refcount zero alone does not establish GPU safety. The initial single-thread design needs no atomic refcounts.

Distinguish allocator IDs/metadata from GPU tensor ownership and byte reservation. Preserve the pure C++ allocator boundary. Use checked wide byte-size arithmetic, generation-aware handles where implemented, and explicit invalid-handle behavior. Check pool/request destruction order and moved-from states.

For growth, examine every possible failure before commit: insufficient pages, metadata capacity growth, copy-on-write allocation, and any asynchronous-copy setup. Existing table, length, and references must remain consistent. Propose the smallest transaction boundary that makes rollback demonstrable.

Share only complete immutable pages under the initial contract. Partial-page writes require an independent tail or a tested copy-on-write protocol. Automatic prefix-cache entries are owners too; do not infer them from ordinary request refcounts.

## Produce evidence proportional to the change

Use focused boundary cases before broad stress: capacity 0/1, exhaust/recycle, P-1/P/P+1 growth, invalid/released handles, non-LIFO completion, shared-prefix release order, cancellation, and failure rollback. A randomized independent model should save seeds and reduce failures. Run relevant host sanitizers when code execution is in scope; otherwise clearly mark proposed commands unrun.

For latency, distinguish O(1) page operations from O(k) request operations and amortized vector growth. Keep heap allocation and debug checks out of claims about a preallocated steady-state path. Compare equivalent work; allocating an ID is not equivalent to allocating/initializing a payload.

Return the invariant or defect, minimal triggering sequence, proposed change/exercise, relevant tests actually run, and any contract dependency. Send proposed log content to the assigned integrator if another worker owns shared documentation. Follow [team ownership rules](../../../docs/TEAM_WORKFLOW.md) only when collaborating; do not launch additional workers just because this role exists.

---
name: kvlab-integration-review
description: "Review changes crossing this project's allocator, GPU cache, metadata, bindings, and scheduler boundaries for lifetime, ordering, failure safety, and contract regressions."
---

# CPU/GPU integration review

Read [project guidance](../../../AGENTS.md), the affected diff and callers, relevant [architecture](../../../ROADMAP.md#2-project-architecture-and-major-component-interactions), and current decision/log records. Review is read-only unless the task includes fixes. Do not expand to a whole-codebase audit without a reason.

Trace one request through reserve, metadata publication, KV write, attention read, completion, cancellation, and reuse. Identify the owner and completion condition for every page, arena tensor, table, pinned staging buffer, event, and output. Check request-slot reuse as well as page reuse.

Key failure modes to test against actual code:

- Refcount reaches zero before a GPU reader completes; generation IDs do not repair an already-submitted stale table.
- A completion event covers one stream but another still reads the page.
- Pinned staging or device metadata is overwritten while an async operation consumes it.
- Length/new table is published before the matching KV append or copy-on-write copy.
- Framework wrapper launches on the wrong stream or loses the tensor owner behind a borrowed pointer.
- OOM, exception, cancellation, or launch failure leaves partial ownership or leaks capacity.
- Shutdown destroys arena/pool/events before submissions drain, or throws from destructors.
- A layout/precision/alias change violates another module's assumptions while local tests still pass.

Examine registration and fake/meta behavior only if custom-operator/compile integration is affected. Do not impose PyTorch machinery on the pure CPU allocator. Distinguish caller validation from internal invariants.

Prefer a concrete delayed-reader, boundary-growth, cancellation, shuffled-page, or cross-stream reproducer to speculative warnings. If execution is available within scope, run focused verification. If not, report the exact untested condition. Treat a correct synchronized implementation as a legitimate baseline; extra overlap is justified by measured control/launch cost.

Return actionable findings ordered by consequence, each with verified file/line, trigger, effect, reasoning, and relevant test. Clearly separate bugs, questions, performance hypotheses, and optional style suggestions. If no findings remain, say what was reviewed and its limits without claiming general proof.

Coordinate shared interface changes through [TEAM_WORKFLOW.md](../../../docs/TEAM_WORKFLOW.md). A review role is not authorization to spawn workers, alter account permissions, push, merge, or rent GPUs.

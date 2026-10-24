# Development and review conventions

This repository is a learning project using practices that scale to larger systems. Apply rigor where contracts, ownership, or performance claims need it; avoid process that exceeds the change. The student remains the primary implementer.

## Start from a contract

Before a substantial change, describe the trigger/problem, supported inputs, outputs, ownership, stream/lifetime rules, failure behavior, and observable acceptance criteria. Link the relevant roadmap phase and any changed design decision. For a small bug, a reproducer and focused explanation are enough.

Keep `kvlab_allocator` independent of CUDA/PyTorch. Keep numerical reference code independent of optimized indexing/reduction code wherever practical. Put tensor validation at public boundaries, and use debug invariants internally. Do not silently expand shapes/dtypes/layouts beyond tested support.

## Keep changes reviewable

- Use one behavioral purpose per branch/PR. Default branch prefix: `codex/` for agent-created branches, unless the user specifies another.
- Separate formatting, refactoring, semantic changes, and performance experiments when separation improves review. Do not mix a scheduler rewrite with a kernel-layout optimization.
- Make changes small enough to reason about; there is no arbitrary line-count limit. Prefer an incremental vertical slice over broad placeholder abstractions.
- Keep the reference and previous correct variant available until the replacement is validated. Use a narrow dispatch/configuration flag where a baseline comparison needs it.
- Respect existing local changes and other contributors' ownership. Inspect the diff before editing; do not reset, overwrite, or reformat unrelated work.

## C++ and CUDA rules that matter here

Use RAII owners and explicit borrowed views. Move-only request ownership is the default. Use fixed-width IDs and checked wide arithmetic for byte sizes. Define ordinary failures such as capacity exhaustion through a clear return/status contract; reserve assertions for internal invariants, not untrusted runtime dimensions. Destructors must not throw; explicit drain/shutdown reports pending GPU errors.

Avoid allocation, logging, or hidden blocking in a path described as bounded-latency. If a `vector` may grow, document it and measure it. Avoid atomics until a cross-thread ownership model requires them. An atomic refcount does not prove a GPU reader is finished.

Public GPU wrappers validate device, dtype, shape, stride, alignment, aliasing, and capacity. Launch on the caller's designated/current stream and keep owners alive through completion. A launch error check and an execution-completion check serve different purposes. Do not fix an ordering bug by adding unexplained device-wide synchronizations throughout the code.

Start with portable scalar operations, clear reductions, and explicit masks. A vectorized path needs alignment/tail tests and a fallback or rejection rule. Fast-math, layouts, precision, and graph replay are distinct experiments with numerical contracts.

## Appropriate verification

| Change | Expected evidence |
|---|---|
| Documentation only | Facts/status/link review; no invented code tests |
| Allocator behavior | Focused regression, invariant/model tests, host sanitizer coverage as relevant |
| Indexing or CUDA math | Numerical oracle cases, boundary/shuffled mappings, appropriate sanitizer smoke |
| Ownership/stream integration | Delayed-reader/reuse/cancellation tests and ordered-work explanation |
| Performance-only change | Correctness rerun, paired baseline/candidate measurements, environment/config/raw data |
| Build/dependency change | Clean affected configuration, working environment manifest, relevant smoke tests |

Run the relevant checks first. Broaden when risk, failures, or new behavior justify it. Repeating the entire GPU matrix after a typo fix wastes both time and budget. A passing formatting check is not correctness evidence; a passing numerical test is not performance evidence.

## Dependencies and reproducibility

Pin third-party source revisions/checksums, Python environment locks, and the selected GPU image digest. Keep local and remote locks separate if architectures differ. Introduce a dependency only when it solves a concrete need; record license and why the standard library or existing dependency is insufficient. Never commit credentials, account keys, generated build trees, or unbounded profiler dumps. Keep compact raw result datasets and selected reports with provenance; archive large traces with recorded paths/checksums under the budget policy.

CPU CI should run formatting/build/tests/sanitizers on relevant supported configurations. Paid GPU CI is manual/budgeted until there is a practical dedicated runner. Fail correctness gates deterministically; review noisy performance distributions instead of making one timing sample a gate.

## Review and merge

The author explains what changed, why, and what was executed. The reviewer challenges correctness, failure safety, assumptions, and measurement fairness. Findings need a concrete scenario, code location, impact, and a suggested verification; style preferences alone should not block a correct focused change unless they violate an established convention.

Resolve material findings and rerun affected checks before integration. The integration owner reconciles API changes, runs cross-module tests, and checks the final diff for unrelated changes. Do not claim a review or CI run occurred when it did not. Publishing, pushing, and merging follow the user's task authorization and repository protections.

PR description outline:

```text
Problem and resulting behavior:
Supported contract / important tradeoff:
Validation executed (commands + result):
Performance evidence (if relevant; run IDs and boundaries):
Known limitations / rollout or revert path:
Documentation / ADR / engineering-log links:
```

Update the engineering log alongside implementation. Record real bugs and rejected hypotheses. Keep comments focused on invariants and reasons; names and code should communicate routine mechanics. Use [the team workflow](docs/TEAM_WORKFLOW.md) when splitting work across people or agents.

## Local-only blog policy

The entire `/docs/blogs/` directory is gitignored, including articles, indexes,
and raw evidence. Maintain it locally, but never stage it, force-add it, or copy
its content into tracked paths. Keep code validation summaries and architectural
decisions in public engineering docs. Do not link ignored files from public
Markdown. Publishing blog material requires separate explicit authorization.

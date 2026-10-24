# Team and subagent workflow

This workflow supports a small team or explicitly requested subagent work while keeping the student's understanding central. Skills are role guidance, not autonomous workers or permission to expand scope. No background agents, GPU jobs, or recurring automation are configured by these documents.

## Roles and focused skills

| Role | Skill | Owns / produces | Boundary |
|---|---|---|---|
| Allocator engineer/reviewer | [kvlab-allocator](../.agents/skills/kvlab-allocator/SKILL.md) | CPU pool/request invariants, rollback, sharing, model tests | Does not change kernel layout or device lifetime contract alone |
| CUDA kernel engineer/reviewer | [kvlab-cuda-kernels](../.agents/skills/kvlab-cuda-kernels/SKILL.md) | Math, indexing, launch/reduction strategy, correctness cases | Does not change benchmark definitions to improve a result |
| Performance investigator | [kvlab-performance](../.agents/skills/kvlab-performance/SKILL.md) | Hypothesis, controlled protocol, evidence, statistical interpretation | Does not claim gains or buy resources without evidence/authorization |
| Integration reviewer | [kvlab-integration-review](../.agents/skills/kvlab-integration-review/SKILL.md) | CPU/GPU contracts, streams, failure/lifetime checks, concrete findings | Review is read-only unless fixes are part of the task |
| Engineering record editor | [kvlab-engineering-record](../.agents/skills/kvlab-engineering-record/SKILL.md) | Log/ADR/experiment provenance, limitations, budget record | Does not invent missing history or silently reinterpret results |

Paths above are repository-local. These skills are supplied as `.agents/skills/<name>/SKILL.md` with metadata. Open the project folder in a future coding session so its project instructions can be read; if a skill is not listed by that session, point the agent to its exact file. Merely placing files here does not prove that the current chat's skill catalog has reloaded.

Use one relevant skill for a focused question. The primary mentor/integrator combines outputs, resolves conflicts, and updates shared records. Do not load all role instructions for a small change. No model override is required by this project.

## Contract-first work breakdown

Before parallel work, freeze the minimum shared contract: tensor shapes/layout, dtype/rounding, page ID and invalid-sentinel representation, length semantics, ownership, stream, completion, failure/rollback behavior, and supported cases. Put changes to that contract in a short decision record. A change to `cache_view.hpp` is a coordination point even if it is only a few lines.

Good independent tasks after the contract exists:

- Review allocator rollback and prepare boundary tests while another reviewer checks RMSNorm semantics.
- Analyze an existing JSONL dataset while an implementer makes a separately scoped CPU fix.
- Review a proposed attention layout without modifying the active kernel files.

Poor parallel tasks: two agents both editing the decoder, simultaneous page-layout changes and binding changes without a frozen ABI, or multiple agents appending to the same engineering log. First reduce coupling or assign a single owner.

## Work order / handoff contract

Give each worker a small self-contained assignment. Use this template only when a multi-person or multi-agent handoff actually benefits from it:

```text
Task ID and outcome:
Relevant phase and current status:
Inputs: exact files/commits/configs/raw artifacts to inspect
Scope: design / review / implementation / measurement
Owned writable paths:
Shared interfaces that must remain compatible:
Supported input and numerical/lifetime contract:
Expected deliverable and acceptance evidence:
Required checks relevant to this change:
Time/compute cost ceiling, if any:
Coordination point if the contract needs to change:
Return: summary, diff/files, tests actually run, findings/risks, raw paths
```

For review tasks, owned writable paths can be empty. For the student's implementation, ask the agent for a test plan, counterexample, or explanation rather than a full solution when that better serves learning. A handoff should never claim that a worker owns a file another worker is editing.

## Isolation and integration

When parallel code editing is explicitly requested, prefer separate branches/worktrees for disjoint tasks if repository instructions and tooling permit. Inspect current worktree/artifact state first and reuse an appropriate checkout. If agents share a checkout, enforce disjoint file ownership and one integration owner; sharing the filesystem is not isolation. Do not run concurrent formatting, builds that rewrite shared generated files, or Git mutations in one checkout.

The integrator owns shared build files, common interfaces, package locks, and the chronological log unless assigned otherwise. Workers return proposed log entries with evidence; the integrator appends them in order. Do not manufacture a large team hierarchy for a solo project.

Integrate one coherent change at a time: inspect worker output, verify it matches the agreed contract, reconcile conflicts semantically, run focused integration tests, inspect the final combined diff, then update the log/ADR. A worker's “passed” statement should identify the exact command/environment/result. On contradictory findings, reproduce the smallest case before choosing an answer.

## Large-codebase practices scaled to this repository

| Practice | Concrete application here |
|---|---|
| Layered dependencies | Allocator has no CUDA/framework headers; launch code accepts explicit views; bindings validate framework objects |
| Stable contracts | Document rounding, page/length conventions, stream ownership, unsupported cases; change them deliberately |
| Ownership map | Maintainer/role table for allocator, kernels, harness, integration, docs; no fabricated CODEOWNERS usernames |
| Small PRs | Separate behavior changes, refactors, and performance experiments; each has a rollback path |
| Regression discipline | Minimal reproducer for each real bug; retain independent reference implementations |
| Reproducible builds | Pin dependencies/image and record architectures/flags; clean-checkout validation before portfolio release |
| Operational visibility | NVTX ranges, structured run IDs, immutable raw results, explicit capacity/error counters |
| Failure design | OOM rollback, bounded admission queue, cancellation semantics, shutdown/drain behavior |
| Evidence-based performance | Same-host paired comparisons and confidence; negative/inconclusive outcomes are valid |
| Knowledge continuity | Session log plus consequential ADRs, not an elaborate document for every variable rename |

## Reviewer responsibilities and severity

Block integration for an observed or well-supported memory-safety, numerical-correctness, data-loss, contract, or budget-accounting defect. Identify the concrete trigger. Distinguish a probable bug from a question, an unmeasured performance hypothesis, and a style suggestion. Scope verification to impact; do not demand a full rental run for a prose change.

Recommended review questions: Can this read invalid KV? Can a pointer/page outlive its owner? Can async work still use a recycled slot? Is length published before data? Is the reduction mask legal? Does failure leave observable state? Does the baseline perform the same operation? Is a claimed service metric actually a layer microbenchmark? Does a reported test have execution evidence?

## Definition of done

A code task is done when its agreed behavior is implemented, relevant executed checks pass, material review issues are addressed, supported/unsupported cases are documented, and the log records the actual evidence. A performance task also requires raw data, controls, uncertainty, and a decision. A documentation task requires accurate links/status and may have no code tests. A GPU task additionally requires exported artifacts and reconciled resource state/cost.

The user retains decisions about expanding scope and spending beyond a released gate. Routine authorized fixes and read-only reviews should proceed without unnecessary permission loops. Skills provide guidance; they do not grant new external-action authority.

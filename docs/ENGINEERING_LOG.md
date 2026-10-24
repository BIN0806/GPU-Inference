# Continuous engineering log

This is the contemporaneous technical record. Keep entries in chronological order. Correct an earlier error with a dated amendment instead of silently rewriting evidence. Link long experiments and code diffs rather than copying every terminal log.

## Entry 0001 — 2026-10-23 — Project design and infrastructure research

**State:** planning completed; implementation not started.

### Objective

Create an executable student-led roadmap for a C++ KV allocator, fused CUDA residual/RMSNorm, and single-layer paged decoding under a $100 total infrastructure limit. Establish the documentation process before implementation.

### Observed starting state

- The supplied folder `/Users/billy/Projects/LLM Kernel Inference ` existed and was empty. Its name includes a trailing space.
- The local machine reported ARM64, Apple Clang 17.0.0, CMake 4.4.4, Ninja 1.13.2, Python 3.14.8, and an Xcode developer directory.
- No applicable ancestor AGENTS.md was found at the checked filesystem/user/Projects levels.
- No C++ project, CUDA kernel, or benchmark was built or executed. No GPU was rented. Spending from this session is $0.

### Design decisions proposed

1. Use a CPU-only allocator of physical page IDs and a separately owned GPU K/V arena. This enables extensive local testing and avoids per-request device allocation.
2. Use one scheduling thread, one stream, and explicit completion before reuse first. Refcounts alone do not protect in-flight GPU readers. Event retirement follows correctness.
3. Start with separate K/V tensors shaped `[physical_pages,kv_heads,tokens_per_page,head_dimension]`; begin at P=16, MHA, D=64/128, FP16 storage.
4. Implement whole immutable page sharing before partial-page copy-on-write or automatic prefix lookup.
5. Match residual rounding and both outputs across fused/unfused/framework baselines.
6. Make end-to-end model integration optional; synthetic layer throughput must never be presented as model generation throughput.

See [ADR 0001](decisions/0001-scope-and-lifetime.md) and the [roadmap](../ROADMAP.md).

### Research problems and resolution

**Problem:** provider pages contain conflicting price text. **Diagnosis:** compared Runpod's pricing table to its GPU product pages. **Resolution:** record primary catalog rates as advertised and require a deployment quote; do not use the inconsistent lower body-text rate. No price was treated as a reservation.

**Problem:** public Runpod deployment inventory was not accessible without sign-in. **Resolution:** mark availability unverified there; inspect public Vast.ai offers instead. Live examples for 3090 and 4090 were visible, but profiling permissions were not demonstrated. See [research notes](RESEARCH_NOTES.md).

**Problem anticipated, not encountered in code:** a cancelled request could release a page while a GPU kernel still reads it. **Proposed solution:** synchronize first; later use retired pages and completion fences. No claim is made that a lifetime bug has already been reproduced or fixed.

### Artifacts produced

Roadmap, README, project guidance, this log, initial ADR, budget ledger, research notes, and experiment template. These are design/documentation artifacts, not implementation evidence.

### Validation performed

Inspected the folder and tool versions; consulted current primary technical/provider sources and public marketplace UI. The documentation package is checked for its twelve required sections, internal file links, budget arithmetic, and consistent status labels before delivery. No CUDA performance or profiling capability was tested.

### Lessons and next action

The scarce resource is focused engineering/debugging time; low hourly GPU prices do not remove the need for preparation. Start with the four-page allocate/exhaust/reuse test locally, then add failure and ownership invariants. Keep GPU expenditure at $0 until Phase 1's gate.

## Entry 0002 — 2026-10-23 — Team development and focused agent skills

**State:** documentation and skill definitions created and structurally validated; project implementation still not started.

### User direction

Extend the roadmap with appropriate skill files for subagents and practices applicable to large codebases and teams, while preserving the student's role as implementer.

### Decision and alternatives

Created five focused project-local skills: allocator engineering, CUDA kernels, performance experiments, integration review, and engineering records. Chose bounded role guidance over one large omnibus skill or a continuously running agent team. Added CONTRIBUTING.md and TEAM_WORKFLOW.md for module boundaries, ownership, contract-first handoffs, small PRs, relevant checks, and one integration/documentation owner.

These files do not start agents or authorize GPU provisioning. When parallel editing is requested later, separate worktrees or disjoint file ownership and an explicit integrator prevent collisions. Shared contracts and log changes require coordination.

### Validation and tooling

Read the skill-creator instructions, used its initializer and validator, and generated UI metadata. The system Python environments did not have PyYAML; a sandboxed download initially failed because network name resolution was unavailable. Installed the pinned validation dependency PyYAML 6.0.3 into a temporary directory using the approved network-capable tool invocation. No project runtime dependency or global Python environment was changed.

The bundled `quick_validate.py` passed for all five skills. Additional checks passed for their UI-description lengths, every relative Markdown file link, Markdown code-fence pairing, all twelve ordered roadmap sections, spending-envelope arithmetic, and the KV-cache size example. Manual review checked role boundaries, mentoring scope, and status/evidence language. These checks validate documentation/skill structure; they do not establish future agent behavior or CUDA correctness. No independent agent forward-test or GPU experiment was performed.

### Next action

Deliver the reviewed planning package to the requested folder, then start the student's four-page CPU allocator exercise in a subsequent development session. Recorded infrastructure spending remains $0.

## Entry 0003 — 2026-10-23 — First implementation and blog installment

**State:** implemented and tested in Debug, Release, and UBSan-only configurations; ASan execution unresolved on this host. Blog and evidence delivered.

### Direction and scope

The user authorized implementation in publishable chunks and requested posts under `docs/blogs`. The first increment implements CPU page allocation/reclamation and reproducible tests. Refcounts, request tables, GPU resources, and performance benchmarks remain future work.

### Design and implementation

Added a C++17 library, public PagePool/PageHandle/ReleaseResult contract, generation checks, a preallocated free-ID stack, statistics, an explicit invariant checker, and a four-page demo. Added CMake presets, test-only pinned GoogleTest, sanitizer options, and focused/model tests. [ADR 0002](decisions/0002-first-page-pool-contract.md) records alternatives and limits. The blog series (local-only record) gives future chunks their own testable boundaries.

### Risks evaluated and observed findings

Stale-ID reuse was an anticipated failure mode: a previously released ID could point to a new live assignment. Generation-tagged handles and a focused regression case protect that transition within the issuing pool's lifetime. This was not a failure discovered in an older implementation. Cross-pool provenance remains a caller precondition. Generation exhaustion fails closed rather than wrapping.

Inspection found that Git resolves the supplied folder to a broader home-directory repository, not an independent project repository. No parent-repository Git mutation, staging, commit, or push was performed. Source hashes are used for this chunk's validation record; establish a dedicated repository root before publishing commits.

The first Debug configure/build/test completed successfully with 11 GoogleTest cases plus one demo CTest. No allocator failure was observed in that initial run. A code review changed archive extraction timestamps to use extraction time, avoiding stale timestamps when a pinned dependency is deliberately updated. Release also passed 12/12. A tests-disabled build configured and ran the demo without fetching GoogleTest. Clean Debug and UBSan-only builds from the delivered project path both passed 12/12, including the directory name with a trailing space. Exact commands, source hashes, outputs, and limits are in the validation record (local-only record).

### Problem encountered: sanitizer startup

The combined ASan/UBSan build compiled, but CTest discovery timed out after five seconds. The sandboxed run also reported a blocked `/bin/ps`. An unrestricted retry removed that diagnostic but still timed out; a direct verbose run exceeded 45 seconds after ASan interceptor initialization. A minimal `std::puts` program reproduced the hang under combined ASan/UBSan, without allocator or GoogleTest code. The same probe and the full project ran successfully with UBSan only. This supports an environment/runtime diagnosis but leaves the exact cause unresolved; ASan is not marked passed. No allocator change was made to mask the symptom. Preserve the failed runs and repeat ASan on a functioning host.

### Cost, learning, and next action

GPU spend remains $0; no resource was provisioned. The first blog post (local-only record) explains invariants, generation handles, container alternatives, reproduction, and the observed tooling failure. A minimal probe narrowed the sanitizer investigation more effectively than changing allocator code without evidence. No performance experiment or speedup is claimed. Proceed to reference-counted complete-page sharing in part 02, carrying the ASan validation item forward.

## Entry template — copy for each session

### Entry NNNN — YYYY-MM-DD — Specific engineering question

**State:** proposed / implemented / tested / measured / unresolved.

**Objective and prediction:** What are we trying to establish? What result would disprove the prediction?

**Alternatives and decision:** Compare at least the relevant simpler alternative; state performance, memory, correctness, and complexity implications. Link an ADR if needed.

**Implementation:** Commit/diff, files/functions, meaningful snippets, and changed invariants. Explain why the change is needed.

**Problem encountered:** Exact symptom, minimal reproduction, seed/shape, expected versus actual result. Distinguish a real failure from an anticipated risk.

**Diagnosis:** Hypotheses, checks, debugger/sanitizer/profiler evidence, rejected explanations, and root cause or remaining uncertainty.

**Solution and verification:** Code change and relevant regression tests, exact commands, exit/results, and limitations. Do not label unrun tests passed.

**Experiment:** ID/config, baseline/candidate, environment, raw-data links, correctness metrics, timing boundary, before/after values with units, uncertainty, and rejected/invalid runs. Use N/A for a non-performance session.

**Cost:** New cash outlay, accrued usage, active resources, cleanup verification, cumulative ledger link.

**Learning and next action:** What is now understood? Keep, revert, simplify, proceed, or investigate—and why?

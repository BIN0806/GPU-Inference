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

## Entry 0004 — 2026-10-23 — Reference-counted sealed pages and ownership leases

**State:** implemented; clean Debug, Release, and UBSan-only builds pass 32 checks. ASan remains unvalidated on this host.

### Contract and prediction

Extend the single-threaded page pool with references starting at one, checked retain/release, and irreversible sealing of complete pages. Predict that owner teardown order will not affect final capacity and that rejecting a mutable-page share leaves state unchanged. A generation-tagged handle identifies an allocation, not an individual owner: manual duplicate release while another reference exists cannot be identified by the count alone. Add a move-only RAII PageLease to balance references for normal callers, while preserving the low-level pool API and its explicit obligations.

### Alternatives and scope

Use non-atomic counters because all metadata operations remain on one owner thread. Sealing requires a caller-declared valid-token count equal to the configured tokens per page; no KV payload exists yet, so the allocator cannot prove a caller wrote those tokens. Once sealed, a page stays immutable even when its count returns to one. Copies of handles do not retain. Sharing retains the existing generation; final release returns the ID, and reallocation starts a new generation. Request tables, automatic prefix lookup, partial-page copy-on-write, and GPU completion fencing remain later chunks.

### Implementation and review

Changed PagePool metadata from a live bit to checked references plus a sealed flag. Added seal/retain/inspect APIs, positive page geometry, and invariant checks requiring all shared pages to be sealed. Added PageLease allocation/sharing, moves, reset/destruction, ownership preconditions, and a shared-prefix metadata demo. Added sharing/lease test modules and a UBSan-only preset; no new dependency. [ADR 0003](decisions/0003-sealed-pages-and-leases.md) explains alternatives and the part 02 article (local-only record) teaches the transitions.

The first Debug run passed 31 checks. Manual contract review identified an additional coverage gap: the lease factory's behavior when retain overflows. Added a controlled counter-saturation case to verify no extra owner is created, then validated the final 32-check suite. No allocator defect or failing regression was observed during these runs; the duplicate-owner-release scenario is an anticipated manual-API limitation, not a fixed runtime failure. Tests also cover all six release orders for three owners, exception unwinding, cross-pool moves, and shared-page move assignment.

### Executed evidence and limitations

Clean builds from the actual project path passed all 32 checks in Debug, Release, and UBSan-only modes. Each suite includes 100 × 10,000 sharing-model decisions and the original 16 × 10,000 pool-model decisions; repeating the same seeds under different modes is not additional independent workload coverage. Exact commands, outputs, and source hashes are in the validation record (local-only record).

The unchanged ASan host-startup limitation was not retried or relabeled as fixed. UBSan is not equivalent memory-error coverage. No GPU code, payload write enforcement, request rollback, concurrency test, or performance measurement is claimed.

Since the project still has no independent Git history, saved a source/build implementation patch. Reversed it in an isolated temporary copy and verified all eight original chunk 01 input hashes; reapplied it and verified all thirteen chunk 02 input hashes. Historical article/evidence annotations now distinguish their snapshots from current code. No parent-repository Git metadata was changed; no staging, commit, push, or publication was performed.

### Learning, cost, and next action

A generation identifies an allocation; a refcount counts owners; a lease balances one ownership obligation; a seal restricts mutation. None of these is a GPU completion fence. The distinction is now reflected in APIs, tests, and documentation. Total infrastructure spend remains $0, with no active cloud resource. Continue to per-request page tables, transactional acquisition, and rollback tests in chunk 03. Keep the first GPU spending gate closed until Phase 1's remaining acceptance criteria are met.

## Entry 0005 — 2026-10-23 — Transactional request page tables

**State:** implemented; final Debug, Release, and UBSan-only builds pass all 52 checks without compiler warnings. ASan remains open.

### Design before implementation

Add a move-only RequestState owning PageLease entries. Separate reserved length from committed length: reservation acquires capacity without declaring data valid; commit declares materialized tokens, seals newly completed pages, and exposes only committed token addresses. Share only a complete committed prefix into an empty request from the same pool. Request construction does not materialize payloads.

Use a temporary candidate table for growth/sharing. Allocate its host metadata before acquiring new references; publish via a nonthrowing swap only after success. On failure, temporary leases clean up; existing request handles/lengths/references stay unchanged. A failed partial allocation may advance generations of briefly acquired pages or change free-list order, which are not rolled back. Candidate construction/movement costs O(total table size) at a page boundary; no bounded-latency claim.

Use C++17 polymorphic memory resources for table metadata so allocation failure is deterministic and isolated in tests. The resource and pool outlive the request. Move construction steals storage; move assignment prepares replacement storage using the destination resource before releasing either side's ownership. Test capacity failure, host bad_alloc, injected generation exhaustion after partial acquisition, and retain overflow after partial prefix sharing.

### Implementation and alternatives

Added RequestState, typed operation statuses, committed-token address lookup, transaction staging, invariant diagnostics, and a runnable request demo. Candidate tables use the same PMR resource as their destination, preserving allocator-equality requirements for swap. Move assignment deliberately may allocate/throw; move construction remains noexcept. Resource behavior was checked against the C++ working draft; [ADR 0004](decisions/0004-request-transactions.md) records the alternatives and cost. Moved the existing pool test seam into one shared header to avoid divergent definitions across translation units; added an isolated failing-resource fixture.

The full candidate design prioritizes a simple publication boundary over allocation efficiency. Crossing a page boundary takes O(total target table size) host work and temporary storage; repeated incremental growth can accumulate quadratic work. An in-place journal or maximum-capacity reservation is a measurement-driven follow-up, not an unmeasured optimization claim.

### Observed problem, diagnosis, and fix

The first Debug suite passed 52 checks but the compiler emitted three unused-result warnings in exception assertions. GoogleTest's EXPECT_THROW executes its statement without consuming the status returned by a nodiscard API. Changed those three test expressions to explicit `static_cast<void>(...)` discards; kept production nodiscard annotations. The clean final configurations emitted no warnings. Initial diagnostics are retained with the evidence. No allocator defect or failing runtime regression was observed; saturation/generation exhaustion/cancellation were deliberate test injections, not naturally encountered production failures.

### Executed evidence

All 52 checks passed in clean Debug, Release, and UBSan-only builds from the actual project path. The complete suite includes the previous 1,160,000 pool/sharing decisions and 64,000 multi-request model decisions. Fixed seeds are repeated across configurations, not counted as independent new workloads. Tested host bad_alloc before page acquisition, failure after a first new page is acquired, failure after a first prefix retain, capacity failure/retry, resource-aware moves, partial tails, cancellation, extreme arithmetic, and address visibility. Part 03 evidence (local-only record) links commands, output, source hashes, and limits; the blog post (local-only record) explains the engineering reasoning.

Reversed the chunk 03 implementation patch in a temporary copy and verified thirteen chunk 02 input hashes; reapplied it and verified twenty chunk 03 hashes. Historical source manifests and raw outputs remain unchanged. No parent-repository Git metadata, staging, commit, push, or publication was performed. ASan's prior startup failure was not retried or relabeled as fixed. No GPU, CI, or performance result is claimed.

### Learning, next action, and cost

Reservation and commitment are different state transitions. Exception safety requires staging both resources and metadata, and defining exactly what rollback preserves. Pool generations are lifetime history, not transactional request state. Continue with synthetic CPU workload timing and memory/waste accounting before choosing a faster metadata strategy. Infrastructure spending remains $0 with no active cloud resource; Phase 1's remaining measurement and sanitizer gates are still open.

## Entry 0006 — 2026-10-23 — Preparing one detailed Git commit per chunk

**State:** reconstructed from verified saved snapshots; fresh isolated Debug
builds pass 12/12, 32/32, and 52/52 checks. No implementation behavior changed.

### Request, alternatives, and scope

The user authorized adding each chunk to BIN0806/GPU-Inference as a detailed
commit. Remote inspection confirmed an empty public repository with no branches.
A single latest-state commit would lose the teaching sequence; reconstructing
from the saved deliveries preserves working increments. The original project
folder resolved to a home-directory Git repository, so history is assembled in
an isolated repository before attaching dedicated Git metadata to the project.
No ancestor-repository staging, commit, or history rewrite is part of this work.

### Preparation and observed tooling issue

Verified all 42, 69, and 99 files against their respective saved delivery
checksums and checked the current project against the final snapshot. Imported
only those files, excluding build trees and generated artifacts. Each commit
message records the contract, alternatives, executed validation, article/ADR
paths, and limitations. The initial commit includes the earlier planning and
team/skill documents; dates are not backdated to imply pre-existing commits.

The connected GitHub commit API requires a parent commit, so it cannot directly
create the initial root commit. The CLI initially could not resolve GitHub
inside the restricted network environment, and its authentication check reported
failure there. Retrying the authorized read-only checks with network access
confirmed the existing account login and empty remote. Use the authenticated
Git CLI for the root history and a normal push; do not change credentials or
force-push to bypass a failure.

### Executed verification and records

Copied each snapshot into a clean isolated source directory, configured Debug
with the checksum-pinned local GoogleTest archive, built, and executed CTest.
All 12/32/52 checks passed without compiler warnings. The three independent
builds ran concurrently; their durations are not performance measurements.
The [history guide](COMMIT_HISTORY.md) links the fresh configure/build/test logs
and provides commands to inspect and build historical commits. Existing raw
logs and source manifests remain unchanged. Publication preparation updates
only current documentation and adds these logs; source/test/build input hashes
remain those of chunk 03.

The initial Git whitespace check flagged blank context lines inside the saved
unified implementation patch. Those leading spaces are patch syntax, not source
whitespace defects. Preserved the raw patch and reran the check on the changes
excluding archived `*-implementation.patch` evidence files; it passed. This
exception does not exclude production sources or current documentation.

### Learning, limits, and next action

A buildable commit is a useful teaching boundary and a reproducibility artifact.
Snapshot hashes let us establish provenance without inventing earlier Git
history. Release and UBSan runs remain historical evidence; ASan, CI, GPU work,
and performance measurement receive no new validation claim. Publishing this
history provisions no compute or storage rental; infrastructure spend remains
$0. Verify the remote head against the prepared tip after the authorized push,
then continue with chunk 04's CPU workload and memory-accounting experiments.

## Entry 0007 — 2026-10-23 — Keep blog content local and out of Git history

**State:** local-only publication policy applied to all three reconstructed
commits; implementation behavior unchanged.

### Request and decision

The user requested that the blog be gitignored and absent from the repository
anywhere. Adding an ignore rule or deleting tracked files in a new tip commit
would leave the posts in earlier revisions. Rewrite the three-commit history
to remove the complete `docs/blogs` subtree, including its evidence, and add
the root-anchored ignore rule in every revision. Keep the local files intact.
The public engineering log and design decisions remain versioned as requested
by the project's continuous-documentation requirement.

### Implementation and verification

Confirmed that the remote advertised only `main`, with no other branch, tag,
or pull-request ref. Constructed an independent repository from each original
revision while excluding the blog subtree. This avoids carrying old blog blobs
or backup refs into the replacement local object database. Updated public docs
to remove links to absent files and contributor/agent instructions to forbid
staging or copying the local posts into tracked files.

Compared each rewritten revision with its original: code, tests, build files,
dependency settings, and other non-Markdown files are unchanged, except for
the intentional ignore rule. Checked every commit tree and reachable object
for excluded paths and original blog blobs, plus the ignore rule in each
revision. Recorded SHA-256 hashes for all 58 local blog/evidence files before
the operation so local preservation can be verified during synchronization.
No code tests were rerun for this documentation/history-only change.

### Publication safety and limits

Use a force-with-lease update bound to the previously observed remote head;
abort if another writer changes it. Ordinary future pushes remain fast-forward.
This user-authorized history removal is distinct from rewriting history merely
to make a failed push succeed. Previous commit IDs change; do not merge old
history back in. GitHub can retain cached old commits and other users can
retain clones; this operation does not claim a platform-wide purge. The
[history guide](COMMIT_HISTORY.md) records the boundary and reproduction steps.
No cloud resource was provisioned, and infrastructure spending remains $0.

## Entry 0008 — 2026-10-24 — CPU workload replay and physical-memory accounting

**State:** implemented and tested; 67 checks passed in Debug, Release, and UBSan.
Two 108-run CPU sweeps are retained; EXP-0002 supplies the current measurement
protocol and public results. No GPU was provisioned.

### Contract and alternatives

The next Phase 1 increment measures CPU metadata operations and synthetic pool
pressure before GPU spending. Add a diagnostic union-of-pages accounting pass
and deterministic event replay. Every owner must be supplied, including the
explicit shared-prefix owner; reject partial inventories instead of presenting
misleading physical-byte totals. Count committed tokens, reservation gaps, tail
slack, cache-only pages, and free pool capacity separately. All KV bytes are
modeled; there is no device or host KV payload allocation.

An event trace is generated before replay with admissions/appends/completions/
cancellations independent of allocator outcomes. Reusing it across policies
makes rejected work visible. This is a slot-based stress test, not a queueing or
LLM serving simulator. Diagnostic replay is separate from individually timed
operations and checks that both passes make identical decisions. Request-table
PMR allocation counters are included in the measured path; output/trace creation
and full invariant scans are excluded. Successful appends with and without
metadata allocation are reported separately.

### Implementation and initial review

Added memory accounting, a pool-identity query on RequestState, a workload
library, a CLI benchmark, output-contract tests, a serial experiment runner,
and optional plots. Initial Debug validation passed all 67 checks. Review then
identified reporting refinements: distinguish exclusively cached pages and
split append timing by metadata allocation, since grouping boundary growth
with within-page appends would obscure the intended experiment. These were
coverage/measurement-design refinements, not observed allocator defects.

### Observed measurement problem and response

The first sweep completed 108 runs, but inspection found a zero-valued timer
control median and many zero event durations. An empty-control histogram later
showed 6,621 zeros and nonzero steps of 41–42 ns. Subtracting timer overhead or
claiming zero-cost operations would be misleading. Preserved EXP-0001, added
zero-sample counts and the full control histogram, and added a whole-trace pass
with only two clock reads. This separates a mean across offered events from
individual operation percentiles. It does not improve the clock's resolution.

Rebuilt the benchmark and reran its output-contract check successfully in all
three build modes. The unchanged full library/model suites had already passed
67/67 in each configuration, without warnings. Repeated the 108-run sweep with
the revised measurement boundary; every capacity result and offered-trace hash
matched the prior experiment. No allocator runtime defect was observed.

### Evidence and limitations

[The experiment report](experiments/0002-cpu-allocator-screening.md) links the
108-run data table, compact latency histogram, source/binary/configuration
provenance, and inspected charts. [ADR 0005](decisions/0005-cpu-workload-measurement.md)
and [the benchmark contract](BENCHMARKS.md) explain alternatives and exact timing
boundaries. Full raw traces and validation logs are retained under ignored local
artifacts; the blog is separately ignored and has not been published.

At P=16, median admission rejection was about 56.5% without sharing and 43.5%
with 90% offered sharing. Request-table allocations increased from 3,106 to
6,492 per trace; more accepted work and explicit sharing transactions both
contribute. No causal split or speedup is claimed. At 0% sharing, mean sampled
tail slack rose from 34.6 slots at P=8 to 280.7 at P=64. One aggregate timing
outlier (535.93 ns/offered event) remains in the results; its cause is unresolved
on this nonexclusive host. Seed repetitions add timing observations, not new
independent capacity workloads. The figures show observed ranges, not confidence
intervals. A font-cache permission diagnostic did not prevent chart rendering;
both generated figures were inspected. Repository review found generated SVG trailing
spaces and CSV CRLF line endings. Normalized only the public copies, verified
unchanged CSV values/XML semantics, and kept the original raw artifacts intact.

The new tests add 32,000 independent-model unshared transitions and 8,000 shared
transitions per full suite. Existing invariants and transactional-failure tests
remain. ASan's earlier host startup issue remains unresolved; no ASan, Linux,
CI, GPU, or end-to-end serving validation is claimed. Formatting was reviewed
manually; clang-format was unavailable. A tests-disabled Release build also
succeeded and ran benchmark help without fetching GoogleTest. A fresh Debug
build from the delivered project directory (including its trailing space) then
passed all 67 checks without compiler warnings. The output is retained in
`artifacts/validation/delivered-debug.txt`.

### Decision, cost, and next step

Keep the correctness-first candidate-table algorithm. Use the observed allocation
counts to design a separate journal/pre-reservation comparison with the same
failure contract, and batch primitive timings to reduce clock granularity effects.
Do not select GPU page geometry from CPU data. Finish the open sanitizer gate
before calling Phase 1 complete. Infrastructure spending remains $0; this
session installed only optional plotting packages in an isolated local environment.


## Entry 0009 — 2026-10-25 — Can the sanitizer gate actually detect a fault?

**State:** implemented and tested locally and on Linux; local ASan initialization remains unresolved.

### Question and decision

Chunk 04 left ASan validation unresolved. Before optimizing request-table growth,
establish whether the instrumentation and host runtime can detect known faults.
The hypothesis is that a separate Linux runtime can validate the unchanged CPU
allocator even while this Mac's ASan probe remains unresolved. A Linux failure
would require diagnosis, not suppression or a claimed pass.

Split ASan and UBSan switches, retain the legacy combined alias, and add a
combined preset. Require Python for test-enabled builds so the CLI contract and
new checks cannot disappear silently. The alternative of paying for a GPU just
to test CPU metadata was unnecessary. [ADR 0006](decisions/0006-sanitizer-validation.md)
records the configuration, trust, and cost tradeoffs.

### Implementation and checks so far

CMake applies compile/link instrumentation through the existing interface target.
A compilation-database audit checks every first-party translation unit. A separate
probe executable exercises a clean allocation, deliberate heap bounds violation,
and deliberate signed overflow; the Python parent checks exact diagnostic classes,
exit behavior, and timeouts. Regression cases reject missing flags, unexpected
flags, disabling overrides, absent compilation entries, generic crashes,
recovered faults, and timeouts. These faults exist only in the test executable.
No allocator or workload algorithm changed.

Fresh local Debug/Release suites passed 69/69; UBSan passed 70/70. All five modes
built without compiler warnings and passed instrumentation audits. ASan-only
and combined clean probes timed out at 20 seconds, preserving the local limitation.
Adding a flushed stderr marker at entry to `main` produced the same bounded
timeout with empty stdout/stderr in both ASan modes. The marker was not reached;
this narrows the symptom but does not diagnose the initialization failure. The
refined UBSan probe and its audit/tooling regressions then passed 3/3. A
tests-disabled Release build ran benchmark help, and the legacy sanitizer alias
built its probe with both compile/link flags and passed its compilation audit.
CMake selected Python 3.14.8; directly invoked scripts used Python 3.14.0 from
the shell. Logs are retained in ignored `artifacts/validation/chunk-05/`. No performance
result is inferred from these checks. The existing model suites still cover
1,264,000 decisions per full execution.

### Hosted execution and cost

Added a five-mode standard Ubuntu CI matrix with bounded jobs, read-only access,
immutable checkout revision, and no cache/artifact uploads. Verified the target
repository is public and Actions enabled; verified the applicable free standard
runner policy. Hosted execution is pending at this implementation stage.
[The validation record](VALIDATION.md) will identify actual runs and source SHAs.
Infrastructure spend remains $0; no GPU rental was provisioned. The initial
$20 GPU checkpoint is unchanged.

### Learning and next action

A configured sanitizer is not a proven detector, and a detector is not a proof
of application correctness. Keep configuration, host runtime, and allocator
behavior as separately observable checks. Complete Linux validation, retain the
local failure, then return to the Phase 1 timing and request-table decision.

### Hosted result and final decision

[Run 40172396157](https://github.com/BIN0806/GPU-Inference/actions/runs/40172396157) completed successfully on source
[`a5041ec`](https://github.com/BIN0806/GPU-Inference/commit/a5041ece6e01c559161c354474239bad22ab4928). Debug/Release each passed
69 checks; ASan, UBSan, and combined each passed 70. Every flag audit and expected
fault preflight passed. Environment: x86-64 Ubuntu 24.04 image `20261023.334.3`,
GCC 13.3.0, CMake 3.31.6, Ninja 1.13.2, Python 3.12.3. Saved full job logs contain
no compiler warnings or unexpected sanitizer findings. Job statuses are complete,
and the run's artifact API reports zero uploads. The source revision and exact
matrix are linked from [VALIDATION.md](VALIDATION.md). No rerun or allocator fix
was needed to obtain the Linux pass.

The independent Linux runtime works, including leak-enabled ASan coverage of
the existing allocator suite. That supports the original hypothesis without
explaining the Mac initialization issue. Kept the latter as unresolved instead
of conflating a host-specific problem with an allocator bug. The evidence update
is documentation-only relative to the tested implementation commit. Cost is $0,
no GPU was provisioned, and no performance experiment was rerun. The next useful
engineering question is still request-table metadata growth and batched primitive
timing; this validation increment alone does not declare all of Phase 1 complete.

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

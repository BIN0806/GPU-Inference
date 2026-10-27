# 0010: Separate native kernel validation from framework readiness

Date: 2026-10-27. Status: implemented; CPU orchestration tested; hardware pending.

## Context

The native unfused control compiles, but the next hardware session still needs
manual build, smoke/full execution, and three sanitizer invocations. The generic
GPU preflight exercises a different tiny kernel and frameworks. It cannot prove
RMSNorm correctness, and a missing Triton installation should not prevent
independent validation of standalone C++/CUDA code.

## Alternatives

- Retain manual commands: simplest code, but easy to miss a failing exit code,
  run an old binary, lose diagnostics, or leave child tools alive on cancellation.
- Expand generic preflight into a framework/kernel/session manager: centralizes
  commands, but mixes unrelated evidence gates and increases coupling.
- Add a small native validation runner using the existing subprocess primitive:
  fresh build, explicit mode, ordered numerical/tool gates, bounded commands,
  and one retained report. Chosen.

## Decision and consequences

Build in a new output directory and require explicit host-only or GPU mode.
Host success always keeps the hardware flag false. GPU success requires complete
34-case smoke output, clean memcheck/racecheck/synccheck smoke runs, complete
218-case full output, and unchanged source/binary hashes. The native checker
owns numerical assertions; the runner owns completeness and failure propagation.
Tool summaries are strict: an unknown format costs an inspection/retry rather
than granting a false pass. Case identities are repeated intentionally as an
external coverage contract; a deliberate matrix change must update both sides.

Use one POSIX process-group execution primitive. Add interrupt cleanup and
partial-output retention there, so preflight and native sessions behave alike.
No daemon, provisioning API, framework dependency, performance dispatch, or
credential management is introduced. CI exercises host-only behavior with an
actual compiler; fixtures test unavailable failure paths without claiming GPU
evidence. SIGKILL/detached descendants remain explicit cleanup limitations.

Revisit only when adding another real native variant or kernel. Keep a selected
implementation in the report then; do not build a generic benchmark framework
before one useful hardware experiment exists. Hardware validation is still the
next gate before fusion, framework binding, or performance claims.

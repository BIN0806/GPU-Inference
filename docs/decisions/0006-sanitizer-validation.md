# ADR 0006 — Independently verify instrumentation and sanitizer runtime

- Date: 2026-10-25
- Status: accepted; local and Linux validation executed (see the validation record)
- Scope: CPU correctness/build validation; allocator algorithms unchanged

## Problem

The original combined ASan/UBSan executable stalled on the local Mac, including
an independent probe. Treating that as an allocator defect would conflate host
runtime initialization with application behavior. Treating a successful compile
or any nonzero probe exit as sanitizer coverage would create false confidence.
The old optional Python integration test could also disappear silently.

## Alternatives

1. Continue with UBSan alone. Cheapest operationally, but it leaves heap lifetime
   and bounds checks unverified under ASan.
2. Rent a GPU machine for CPU sanitizer testing. It could provide Linux, but
   spending the GPU budget does not improve this CPU-only experiment.
3. Split sanitizer configurations, isolate runtime probes, and run the existing
   suite on a standard public-repository Linux CI runner. Additional build/test
   plumbing buys an independent environment and repeatable regression evidence.

## Decision

Choose option 3. Keep separate ASan and UBSan options, a combined preset, and the
legacy combined switch. Use target-scoped compile/link options rather than global
flags. Require Python 3.9+ for tests, not for production libraries. Audit all
first-party compilation entries. Require a clean probe to run normally before
fault probes; accept faults only with a matching diagnostic and nonzero exit.
Make all process waits bounded and preserve raw outcomes.

CI uses all five build modes, read-only repository permissions, an immutable
checkout revision, a standard Ubuntu runner, and no Actions caches/artifact
uploads. Gate execution on public visibility and bound job duration/concurrency.
This follows the [verified public-runner pricing boundary](../BUDGET.md), not an
assumption that every GitHub runner is free. No repository protection setting
is modified by this decision.

## Consequences and revisit conditions

Builds with tests now need Python. Existing cached global sanitizer flags must
be cleared explicitly. GoogleTest itself is not fully instrumented; all project
C++ targets are. Additional source directories or conditional source files need
a reviewed audit rule so omitted translation units cannot silently pass.

A Linux pass establishes coverage on that toolchain; it does not fix or explain
the local Mac's runtime problem. Keep the unresolved result. Revisit the CI
matrix if a failure is compiler-specific, the repository becomes private, or
GPU bindings add new ownership boundaries. Device instrumentation belongs in a
separate Compute Sanitizer protocol, not these host flags. See the
[validation contract](../VALIDATION.md) for commands and actual evidence.

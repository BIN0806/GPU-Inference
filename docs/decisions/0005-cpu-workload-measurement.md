# ADR 0005: complete-owner accounting and paired CPU workload replay

Date: 2026-10-24. Status: implemented and tested; local measurements recorded.

## Problem

Per-request rounded capacity double-counts shared prefixes. Pool occupancy alone
cannot distinguish useful tokens from reservations, tail slack, or cache-only
ownership. Likewise, a timer around a short allocator operation can mostly
measure clock granularity. We need defensible evidence before changing the
transactional candidate-table implementation or renting a GPU.

## Alternatives

- Sum request lengths/rounded page counts: cheap, but not physical accounting.
- Maintain counters on every production transition: cheap snapshots, but broad
  invasive changes and another failure-atomicity obligation before measurement.
- Scan a complete owner inventory: slower diagnostics with explicit reconciliation.
  Choose this now; keep it outside timed regions.
- Generate workload decisions from current capacity: easy to implement, but
  changes the offered workload across policies and hides rejection costs.
- Precompute events independently of state: choose this; expose skips, rejected
  admissions, and rejected growth separately. Accepted work may still differ.
- Interleave memory scans with timing: simpler, but scans perturb the following
  operation's caches. Choose a separate diagnostic replay with outcome matching.
- Report only individual nanosecond samples: useful for grouping operations, but
  the host produced zeros and coarse steps. Retain these samples and their
  quality indicators; add a separately timed whole-trace replay.

## Decision and implications

Use `account_memory`, requiring all owners and one optional explicit cache owner.
Fail on incomplete reference coverage, foreign owners, or arithmetic overflow.
KV byte fields are modeled, never described as actual device allocations. The
workload library stays separate from the production allocator target; the latter
adds only the accounting diagnostic and an immutable pool-identity query.

Hold pool token slots constant while sweeping page size. Enable sharing with a
persistent prefix owner and include its cost. Count PMR calls/bytes for request
tables; they perturb timing and do not represent all process allocations.
Public code/results are versioned; articles remain in ignored local storage.

Whole-trace means and individual event percentiles answer different questions.
Never turn replay means into a p99, infer zero-cost work from zero ns, or call a
simulation step LLM decoding. No allocator optimization is made in this chunk.

## Revisit

Add incremental counters only if diagnostic cost blocks useful experiments.
Add a bounded journal/pre-reserved request table only with equivalent failure
semantics and paired measurements. Add a real arrival/queue/service model only
when its semantics are explicit. GPU ownership still needs stream completion
fences; CPU counters and page reclamation do not establish device safety.

See [benchmark contract](../BENCHMARKS.md) and [engineering log](../ENGINEERING_LOG.md).

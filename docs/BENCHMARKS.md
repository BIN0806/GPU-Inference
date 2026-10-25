# CPU allocator experiments

This harness measures single-threaded CPU metadata. It allocates no KV payload,
runs no CUDA, models no queue or service time, and generates no model tokens.
Its purpose is to expose page-size, sharing, capacity, and host allocation
tradeoffs before the first GPU rental.

## Build, test, and run

```sh
cmake --preset release
cmake --build --preset release --parallel 4
ctest --preset release
./build/release/allocator_bench --output artifacts/smoke \
  --events 20000 --seed 7 --pool-tokens 4096 --page-tokens 16 \
  --slots 32 --max-tokens 1024 --prefix-tokens 128 --share-percent 50
python3 scripts/run_allocator_experiments.py \
  --binary build/release/allocator_bench --output artifacts/experiment
```

Output directories must be new; existing results are never overwritten. The
runner requires an unsanitized Release build, launches experiments serially,
and randomizes their order with a recorded seed. Defaults cover 4 page sizes ×
3 sharing settings × 3 trace seeds × 3 fresh-process repeats = 108 runs. Each
run offers 20,000 events. The C++ CLI accepts `--help`; Python uses `--help` too.
The runner uses only the standard library. Optional charts use Matplotlib:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-plot.txt
.venv/bin/python scripts/plot_allocator_experiments.py artifacts/experiment
```

Raw output lives under ignored `artifacts/`. Record compact experiment results
in public engineering docs; blog posts and their own evidence remain local-only.

## Workload contract

A precomputed trace has 25% admission, 50% append, 12% finish, and 13%
cancellation draws. Each event independently selects one request slot. Prompt
lengths are 1..max_tokens; append lengths are 1..16, capped at max_tokens during
replay. Random values use standard `mt19937` raw outputs modulo a range, with
small modulo bias. The exact trace is saved, not reconstructed from timing.
The generator is deliberately simple synthetic pressure, not production traffic.

Admissions targeting an occupied slot and non-admissions targeting an idle slot
are skipped. Appends at maximum length are skipped. Report these separately
from admitted/completed/cancelled/rejected work. Admission rejection fraction
uses admitted plus admission-rejected events as its denominator; busy-slot
admissions are excluded. A failed admission rolls back
any acquired prefix references. A failed append preserves the active request
for a later retry or cancellation. No queue is invented for rejected requests.

When sharing is enabled, a persistent prefix owner reserves the complete-page
part of prefix_tokens before arrivals. Eligible admissions share at most the
complete prefix pages that fit their prompt; their tail is private. Synthetic
prefix equivalence is assumed, not discovered by hashing actual tokens. The
prefix owner consumes capacity even when no request uses it. With sharing
disabled there is no prefix owner allocation. This cache cost is intentional
and included in the paired policy experiment. `drain()` is terminal and releases
all request and prefix references; page-table vector capacity may remain until
destruction. No deferred GPU reclamation or automatic cache eviction exists.

The same seed generates identical offered events across page sizes and sharing
settings. The runner verifies that property with a hash excluding outcomes and
timing. Accepted work can differ after capacity failures; never compare total
runtime as though every policy completed identical work.

## Accounting from first principles

`account_memory` requires the complete inventory of RequestState owners. It
checks pool identity, request invariants, physical page uniqueness, and exact
reference-count reconciliation. An omitted external lease or cached owner is
an error. The optional single cache-owner designation marks pages with no other
owner as `cached_only_pages`. It is an explicit prefix model, not a cache service.

For each distinct page, take the maximum committed and reserved token coverage
across its owners. Complete shared pages are counted once; private partial tails
retain their own coverage. With P tokens per page:

```text
pool_pages = free_pages + live_pages
live_capacity_tokens = live_pages * P
live_capacity_tokens = unique_committed_tokens
                     + reserved_uncommitted_tokens + tail_slack_tokens
shared_capacity_saved_tokens = (logical_page_references - live_pages) * P
bytes_per_token = 2 * layers * kv_heads * head_dimension * element_bytes
modeled_pool_bytes = pool_pages * P * bytes_per_token
```

`live_pages` includes cache-only pages. `shared_capacity_saved_tokens` compares
duplicating the current owners' pages against their physical union, including
the cache owner. It is not the difference between two entire replay policies.
The diagnostic does not time its scans, allocate payloads, or measure RSS.
Every sum/product is checked. A pool that could represent huge metadata lengths
may still overflow its hypothetical byte model and is rejected.

For P=4, reserve 9 tokens and commit 5: three physical pages hold 12 slots,
partitioned into 5 committed + 4 reserved/uncommitted + 3 tail slack. Free pages
are separate from those unused live slots. This distinction avoids calling
intentional reservations or a mostly empty pool fragmentation.

The CLI holds pool token slots constant across P (4096 by default). Its byte
model is one layer, 8 KV heads, head dimension 64, 2-byte elements: 2048 bytes
per token and a hypothetical 8 MiB pool. No such 8 MiB tensor is allocated.
Memory CSV samples are indexed by offered event count, not wall time; means
and peaks are sampled statistics, and transient peaks can be missed.

## Three replay passes and timing limits

1. Warm up up to 2048 offered events in a separate workload, then destroy it.
   This is warmup policy, not evidence of stable clocks or a warmed maximum size.
2. Measure 10,000 pairs of consecutive `steady_clock::now()` calls and preserve
   their histogram. Do not subtract this noisy control from individual events.
3. Individually time each `apply()` in a fresh workload. Metadata allocations,
   releases, PMR counters, dispatch, and contract checks are included. Trace
   generation, result-vector allocation, filesystem output, and full invariant
   scans are outside the interval. Allocation-delta sampling and result writes
   are outside each event timer but inside the overall instrumented loop timer.
4. Replay with diagnostics separately, checking every event outcome against the
   timed pass and sampling complete accounting every 128 events plus endpoints.
5. Replay a third fresh workload with just two clock reads around the whole
   loop. Include dispatch, PMR counters, and an outcome checksum. Verify final
   committed count and checksum; verify teardown. Divide by all offered events,
   including skips and failures. This is an aggregate mean, never a latency p99.

The clock is monotonic, but its actual observable granularity need not be one
nanosecond even if durations are represented in ns. The first local run produced
many zero durations and roughly 41–42 ns steps. Zero means unresolved at this
measurement scale, not zero-cost work. Preserve zero samples, report their count,
and use aggregate replay time to study whole-trace cost. Individual allocation
timings remain instrumented, quantized observations. See the
[C++ steady-clock contract](https://eel.is/c++draft/time.clock.steady) and
[random engine specification](https://eel.is/c++draft/rand.eng.mers).

Per-event groups distinguish operation, outcome, and whether a PMR allocation
occurred. Nearest-rank p50/p95 use individual observations. p99 is null below
10,000 samples in that group; more samples alone would not remove timer limits
or correlation. The instrumented loop rate includes timer/result bookkeeping.
Neither it nor aggregate events/s is LLM tokens/s or request completion latency.

## Artifacts and experimental review

Each run writes `events.csv`, `memory.csv`, `timer-control.csv`, and `summary.json`
(schema 2). The full runner additionally preserves commands, job order, trace
hashes, binary/source hashes, CMake cache, tracked source diff, environment,
pre-run load averages, per-process stdout/stderr, and `runs.csv`. Source hashes
include untracked implementation files; the tracked diff alone is insufficient
to reproduce an uncommitted build. Failed runs retain their logs and failed
manifest status. A binary's Release flag cannot prove correctness or an idle host.

Graphs show medians and observed min–max ranges across seeds/repeats, not
confidence intervals. Multiple repeats of one deterministic trace add timing
observations, not independent capacity workloads. Inspect admissions, skips,
and cancellations alongside latency; otherwise a configuration that rejects
more work can look misleadingly fast. PMR statistics count only RequestState
tables, including prefix ownership and temporary growth candidates. They omit
pool vectors, trace storage, diagnostic scratch, and unrelated process memory.

CLI safety bounds are 5,000,000 events, 16,777,216 pool/max token slots, and
100,000 request slots; these are caps, not guarantees against host OOM. Positive
page size must divide pool slots. Metadata and timing depend on page size;
these results establish no device-memory locality, attention bandwidth, launch
overhead, allocator speedup over another implementation, or serving throughput.

Next experiments should isolate batch-timed page primitives and compare an
alternative request-table growth strategy under equivalent semantics. Repeat
ASan on a working host before declaring Phase 1's sanitizer gate complete.

## Comparing request-growth implementations (chunk 06)

`compare_allocator_variants.py` runs adjacent baseline/candidate processes in
randomized order, with a randomized order of workload pairs. It accepts two
Release binaries plus their adjacent `CMakeCache.txt` and `compile_commands.json`.
The compilation database must contain no sanitizer flags. Both executables must
use the same schema-2 C++ harness. The baseline is preserved at commit `7e82018`;
keep its executable separate so rebuilding the candidate cannot replace it.

```sh
python3 scripts/compare_allocator_variants.py \
  --baseline /path/to/frozen-baseline/allocator_bench \
  --candidate build/release/allocator_bench \
  --baseline-source-manifest /path/to/baseline-source-sha256.json \
  --output artifacts/request-growth-comparison
```

The source manifest maps repository-relative files to SHA-256 hashes from the
baseline checkout. Create it when building that binary and freeze it alongside
the cache/compilation database. The runner records the candidate source hashes,
both executable hashes, source diff, tool configuration, exact schedule, commands,
and pre-run load averages. These record the supplied build provenance; a source
manifest alone cannot establish that an arbitrary executable came from it.

Defaults: 3 page sizes (8/16/64), 2 offered sharing percentages (0/90), 3 seeds
(7/17/29), 3 repeats, 20,000 events: 54 pairs / 108 fresh-process runs. Pool slots
remain 4096 tokens, 32 request slots, max request 1024 tokens, prefix 128 tokens,
and sampling every 128 events. Each child has a 120-second limit. Failure logs
and incomplete manifests are retained; output directories must be new.

For each pair, require matching configuration, outcome counts, every offered
event and its outcome, and the entire sampled physical-memory CSV. Timing and
host PMR counters may differ. Save allocation calls, requested/peak/retained host
bytes, and aggregate ns/offered event in `pairs.csv`. No timing threshold is a
CI gate. Inspect paired ratios and absolute times with all repeats/outliers;
seeds/repeats are not independent production traffic or service latency samples.
Host resource counters remain inside the measured replay in both variants.

Geometric table growth can add physical pages without allocating host storage.
Accordingly the older single-variant runner now calls its append groups
`append_with_metadata_allocation` and `append_without_metadata_allocation`.
Historical CSVs retain their original column names; the C++ raw schema remains
unchanged. An absence of PMR allocation no longer implies "within one page."

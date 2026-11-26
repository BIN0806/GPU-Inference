# Development plan after the native fusion benchmark

The C++ allocator owns page metadata and shared immutable prefixes. The native
CUDA library now has two validated residual + RMSNorm implementations. On one
L4, warm FP16 fusion won at widths 128 and 1024 and lost at 4096. The experiment
records both timing boundaries and retains the two-kernel control.
[Allocator code map](README.md) · [Native comparison](../experiments/0004-l4-fused-rmsnorm.md).

## Next implementation chunk

Give allocator pages actual K/V payloads. Start with a CPU address oracle.
Use one layer, one page size, matching query/KV head counts, and one serialized
stream for the first GPU test.

1. Spell out a single contiguous arena layout per K or V tensor, e.g.
   `[physical_page, token_in_page, kv_head, channel]`. Check products in
   wide arithmetic before allocating. For P=4 and table [7,2], token 5 must
   land on physical page 2, slot 1.
2. Drive page allocation and table changes through the existing request API.
   Build independent expected K/V values keyed by logical token, head, and
   channel. Poison unused physical slots and deliberately shuffle page IDs.
3. Allocate arenas once. Stage request lengths and page-table IDs, append
   on the same stream, and copy gathered values back for comparison. Keep
   table storage and pages alive until the stream completes.
4. Exercise lengths P−1, P, and P+1. The append reaching P fills the existing
   page. The following append needs another page. Check every valid element,
   the unused tail, and a second request with different physical mappings.

Close this increment when the CPU oracle and GPU append agree across that
boundary, including after a page has been released and safely reused in the
serialized driver. Then connect the contiguous attention reference and
single-layer paged decoder. A separate RMSNorm follow-up can investigate the
width-4096 regression with one controlled mapping change.
[Paging roadmap](../../ROADMAP.md#phase-3--paged-attention-decoding).

## Milestones and evidence

| Increment | Working artifact | Evidence that closes it |
|---|---|---|
| Fused RMSNorm | Native fused and unfused variants with the same public contract | Both match the oracle. The fused smoke matrix passes the three sanitizer modes on the L4. |
| RMSNorm experiment | Reproducible native comparison, followed by eager PyTorch, Triton, and `torch.compile` where the environment supports them | Warm each variant, compare the same inputs and outputs on one device, save raw timings, and explain which shapes benefit or regress. Record framework readiness separately. |
| KV storage and append | CPU address oracle, persistent GPU K/V arenas, and allocator-derived page tables | Shuffled page IDs and poisoned unused slots expose wrong mappings. Appends cross `P-1→P→P+1` and the gather result matches the CPU oracle. |
| Single-layer paged decode | Contiguous reference, paged decode, and a serialized driver using real allocator mappings | Outputs agree on variable lengths and shared-prefix cases. Same-stream append, decode, and release work under cancellation and pool pressure. Trace teardown recovers all pages. |
| Lifecycle and systems experiments | CUDA event-based retirement, a synthetic workload replay, and a reproducible report | Delayed-reader tests prove page reuse waits for the event. Compare contiguous and paged paths at matched work and, separately, under a fixed memory budget. Record latency, throughput, page waste, queueing, and metadata transfer cost. |

The first FP16 timing sweep is complete. A comparison with eager PyTorch,
Triton, and `torch.compile` will need a framework-ready GPU environment and
the same rounded two-output contract. Counter access remains unverified.
[Benchmark protocol](../../ROADMAP.md#7-benchmarking-and-profiling-methodology).

For paging, get the address translation, append, and simple attention reference
working before fusing decode or overlapping streams. Start with one layout,
one page size, and matching query/KV head counts. Add a second page size once
the baseline is stable. Keep the allocator responsible for logical ownership
and use a GPU completion event to decide when physical storage can be reused.
Preserve the serialized path for comparison. [Paged decode roadmap](../../ROADMAP.md#phase-3--paged-attention-decoding).

## Expected end result

**Minimum portfolio release.** A clean clone builds and tests the CPU allocator.
A documented GPU command reproduces the validated fused/unfused normalization
comparison. A small synthetic, single-layer decode driver uses the allocator's
actual page tables and matches contiguous attention across page boundaries and
varying request lengths. A compact report and raw artifacts let a reviewer
trace those claims to a revision, hardware configuration, and command.

**Strong portfolio release.** Add the delayed-reader demonstration, integrated
workload replay, and plots that show how page size and prefix reuse change
capacity, latency, and throughput. Explain one confirmed bottleneck or a
well-controlled optimization that failed to pay off. The demo should let you
walk a reviewer through an append, a page-boundary decode, and safe page reuse.
The core release uses synthetic Q/K/V tensors. Loading a real model is an
optional follow-up with its own correctness and budget decision.

Use the existing $20 initial checkpoint for the normalization work and keep
the project within the $100 total ceiling. Plan GPU calls around a concrete
experiment, export the evidence, and reconcile the stopped resource and cost.
The first L4 session's workspace snapshot showed about one cent metered and
$0.00 billed. Estimates for later jobs depend on measured usage and available
credits. [Budget ledger](../BUDGET.md) · [Experiment template](../templates/EXPERIMENT.md).

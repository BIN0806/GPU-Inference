# Development plan after the first L4 run

The C++ allocator owns request metadata, page references, and shared immutable
prefixes. The unfused CUDA residual add and RMSNorm control passed 34 smoke cases,
218 full cases, and three Compute Sanitizer smoke runs on one Modal L4. That
gives us a reference for the fused kernel. The first session established
correctness on this device. GPU timing and profiler access remain open.
[Allocator code map](README.md) · [L4 evidence](../validation/0010-modal-l4.json).

## Next implementation chunk

Add a separate fused entry point to the native CUDA library and keep the
unfused entry point callable. Both produce the rounded residual `z` and the
weighted normalization output `y`. Match the existing dtype, shape, epsilon,
stream, and buffer requirements.

1. Write down the dataflow for one odd-width FP16 row before editing the CUDA.
   Use the existing rounding witness to explain why the sum of squares must
   use the values represented by `z`.
2. Start with one block per row and the same FP32 shared-memory reduction as
   the control. The first pass writes `z` and reduces the rounded values. The
   second pass rereads `z` to produce `y`. This isolates removing a launch and
   one logical read of `z`. Register retention, warp shuffles, and vector
   loads can each be tested later as separate changes.
3. Extend the native checker to select the kernel explicitly and run the same
   independent oracle and case matrices for both. Exercise the API boundary
   checks, FP16 rounding witness, short and odd widths, and empty batches.
4. Compile both target architectures in Linux CI. For the next authorized
   Modal run, record the exact candidate revision and selected implementation.
   The existing adapter pins the already-tested control at `48ed530`, so the
   candidate needs an explicit revision update. Save both numerical reports
   and run memcheck, racecheck, and synccheck on the fused smoke suite.
   [Native contract](../CUDA_RMSNORM.md) · [Modal guide](../MODAL.md).

The chunk is ready when the fused candidate passes the numerical matrix with
the existing tolerances and bitwise residual checks, reports clean sanitizer
smoke results, and has a short explanation of the rounding and synchronization.
Use `kvlab-cuda-kernels` for the implementation and `kvlab-integration-review`
for the stream and buffer boundary. [Team workflow](../TEAM_WORKFLOW.md).

## Milestones and evidence

| Increment | Working artifact | Evidence that closes it |
|---|---|---|
| Fused RMSNorm | Native fused and unfused variants with the same public contract | Both match the oracle. The fused smoke matrix passes the three sanitizer modes on the L4. |
| RMSNorm experiment | Reproducible native comparison, followed by eager PyTorch, Triton, and `torch.compile` where the environment supports them | Warm each variant, compare the same inputs and outputs on one device, save raw timings, and explain which shapes benefit or regress. Record framework readiness separately. |
| KV storage and append | CPU address oracle, persistent GPU K/V arenas, and allocator-derived page tables | Shuffled page IDs and poisoned unused slots expose wrong mappings. Appends cross `P-1→P→P+1` and the gather result matches the CPU oracle. |
| Single-layer paged decode | Contiguous reference, paged decode, and a serialized driver using real allocator mappings | Outputs agree on variable lengths and shared-prefix cases. Same-stream append, decode, and release work under cancellation and pool pressure. Trace teardown recovers all pages. |
| Lifecycle and systems experiments | CUDA event-based retirement, a synthetic workload replay, and a reproducible report | Delayed-reader tests prove page reuse waits for the event. Compare contiguous and paged paths at matched work and, separately, under a fixed memory budget. Record latency, throughput, page waste, queueing, and metadata transfer cost. |

For the first timing sweep, use FP16 and screen a few shapes such as row counts
`1, 32, 128` and widths `128, 1024, 4096`. Compare warmed, preallocated
variants in interleaved trial groups and record CUDA-event time and host
dispatch-to-completion time separately. If cache residency could explain the
result, add a larger working set as a follow-up. Include compilation and setup
in a separate report field. Use a brief Nsight probe to check counter access.
CUDA events and host timing still support the comparison when counters are
unavailable. [Benchmark protocol](../../ROADMAP.md#7-benchmarking-and-profiling-methodology).

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

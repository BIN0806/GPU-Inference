---
name: kvlab-performance
description: "Design or audit this project's CPU/GPU benchmarks, interpret raw measurements, and investigate performance regressions with fair baselines and budget-aware experiments."
---

# Performance experiments

Use [ROADMAP.md section 7](../../../ROADMAP.md#7-benchmarking-and-profiling-methodology), the [experiment template](../../../docs/templates/EXPERIMENT.md), and current [budget ledger](../../../docs/BUDGET.md). Read raw data/configs/environment before accepting a chart or claimed speedup. Preserve mentor-first project guidance.

## Define the question and boundary

Separate GPU operator time, host dispatch-to-ready time, integrated decode-step time, synthetic request delay, and real-model TTFT/inter-token latency. Layer token-steps are not model-generated tokens. Specify which allocations, transfers, compilation, packing, synchronization, and input generation are inside the measurement.

Compare equal numerical/output contracts on the same device/configuration. For paging, distinguish already-contiguous attention, already-paged gather-plus-attention, and fixed-byte-budget admission experiments. Count conversion work where it is required by the application. Record actual framework/SDPA backend and graph behavior.

Warm compilation/module loading/autotuning and clocks separately from measured steady state. Use events on the actual stream and a completion wait; recognize CPU launch gaps in very short event-timed loops. Separate graph replay and native-launch results. Reset mutable inputs. Compare warm-cache and streaming conditions when cache residency could explain the result.

Use independent sample groups/runs and interleaved variants. Do not turn grouped means into per-invocation p99 or treat correlated launches as independent confidence samples. Preserve outliers/invalid-run reasons. With insufficient data, return an inconclusive result and the cheapest discriminating next experiment.

## Interpret evidence honestly

Compute logical byte/FLOP models from the implemented algorithm; do not label modeled effective GB/s as measured DRAM bandwidth. Reconcile unique live, retired, cached, and free pages with arena/framework/scratch reservations. Shared logical references must not double-count physical bytes. Analyze capacity benefits separately from kernel latency.

Use unprofiled timings for reported latency. Nsight replay/counter data explains mechanisms; it is not interchangeable with ordinary execution. Without counters, use events, host timing, available traces, compiler resources, and controlled sweeps; label bottleneck conclusions as inferences when appropriate.

Before any authorized rental experiment, estimate all-in cost, inspect current released budget, prepare a bounded run, and plan verified export/cleanup. This skill does not itself authorize provisioning. Never assume catalog inventory or profiler permissions.

Return absolute measurements with units, baseline/candidate, uncertainty, correctness evidence, metadata/raw links, included/excluded work, limitations, and keep/revert/investigate decision. No fabricated speedups, projected resume achievements, or silently discarded regressions.

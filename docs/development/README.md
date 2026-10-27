# Development guide

Use this guide to navigate the implemented components. The
[roadmap](../../ROADMAP.md) describes the larger target system; the
[implementation history](../COMMIT_HISTORY.md) identifies independently buildable
increments, and the [engineering log](../ENGINEERING_LOG.md) records decisions
and checks as work progresses.

## Allocator code map

| Module | Responsibility |
|---|---|
| [PagePool](../../include/kvlab/page_pool.hpp) / [implementation](../../src/allocator/page_pool.cpp) | Generation-tagged page handles, checked reference counts, sealing, and physical-page accounting. |
| [PageLease](../../include/kvlab/page_lease.hpp) / [implementation](../../src/allocator/page_lease.cpp) | One owned reference per move-only lease. The pool must outlive the lease. |
| [RequestState](../../include/kvlab/request_state.hpp) / [implementation](../../src/allocator/request_state.cpp) | Reserve capacity, commit token counts, share complete immutable prefixes, and translate logical token addresses. |
| [Memory accounting](../../include/kvlab/memory_accounting.hpp) | Reconcile owners, unique pages, reservation gaps, cache-only pages, and modeled KV bytes. |
| [Workload replay](../../include/kvlab/workload.hpp) / [benchmark CLI](../../bench/allocator_bench.cpp) | Seeded offered events, capacity failures, host allocation counters, and reproducible output. |

Request-table growth reuses host storage and rolls back a newly acquired suffix
on failure. Logical state can roll back while successfully grown host capacity
remains allocated. The [growth decision](../decisions/0007-request-table-growth.md)
and [paired experiment](../experiments/0003-request-table-growth.md) explain the
allocation savings, retained-memory cost, and measured regression in one regime.

Request commitment declares materialized tokens; it does not write KV payloads
or wait for a GPU. Borrowed request handles must not be manually retained,
released, or sealed. The pool and any supplied metadata memory resource must
outlive the request. CPU reference count zero does not prove GPU completion.

## Examples and tests

Build using the main README, then run these examples from the repository root:

```sh
./build/debug/page_pool_demo
./build/debug/shared_prefix_demo
./build/debug/request_state_demo
```

The [page-pool example](../../examples/page_pool_demo.cpp) introduces allocation
and reuse. The [sharing example](../../examples/shared_prefix_demo.cpp) shows
sealed pages and ownership. The [request example](../../examples/request_state_demo.cpp)
adds reservation, commitment, and prefix sharing.

Before changing request ownership, explain why reservation does not expose
readable tokens, what rollback preserves, and why page generations can advance
during a failed attempt. Read the [failure tests](../../tests/request_state_test.cpp)
and [multi-request model](../../tests/request_model_test.cpp) alongside the
implementation. Current test counts, sanitizer behavior, CI runs, and platform
limitations belong in [the validation record](../VALIDATION.md).

## Numerical reference code

- [Independent oracle](../../python/kvlab/oracle.py): explicit storage rounding and a tensor-independent high-precision normalization calculation.
- [PyTorch reference](../../python/kvlab/reference.py): checked inputs and separate prevalidated math, returning both residual and normalized outputs.
- [Correctness cases](../../python/kvlab/correctness.py) and [matrix runner](../../scripts/check_rmsnorm_reference.py): seeded inputs, error metrics, source hashes, and failure context.
- [GPU preflight runner](../../scripts/gpu_preflight.py): tool inventory and prepared execution/profiling probes for an existing GPU host.

Follow [the RMSNorm contract](../RMSNORM.md) before implementing a kernel. In
particular, normalization observes the residual after storage rounding. The
[standalone CUDA control](../CUDA_RMSNORM.md) now implements separate add and
RMSNorm kernels. Validate those on hardware before fusion or framework bindings. Device execution and profiling
remain unverified until tested on NVIDIA hardware; see [the preflight guide](../GPU_PREFLIGHT.md).

## Run a CPU experiment

```sh
cmake --preset release
cmake --build --preset release --parallel 4
./build/release/allocator_bench --output artifacts/first-run --events 20000
```

The output directory must be new. Read [the benchmark contract](../BENCHMARKS.md)
before interpreting results: zero-duration samples indicate timer resolution
limits, whole-trace means include skipped/rejected work, and modeled KV bytes
are not device allocations. Keep performance builds separate from sanitizer
builds. Detailed dependency, sanitizer, and migration instructions are in
[setup](../DEPENDENCIES.md) and [validation](../VALIDATION.md).

## Maintain the engineering record

Use [the history guide](../COMMIT_HISTORY.md) for chunk boundaries and past
revisions, [the log](../ENGINEERING_LOG.md) for implementation reasoning, and
[the budget ledger](../BUDGET.md) for resource and spending decisions. Avoid
copying their changing status into the main README. Follow
[contributing conventions](../../CONTRIBUTING.md) when updating code or docs.

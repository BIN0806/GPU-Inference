# LLM Kernel Inference

A C++/CUDA inference-components laboratory built in small, explained, testable increments: a fixed-page KV allocator, fused residual/RMSNorm, and single-layer paged decode attention.

**Implemented: chunks 01–07, CPU ownership/workloads and residual/RMSNorm references.**
Generation-tagged pages, checked sharing, move-only leases, and transactional
request tables now have complete-owner memory accounting and seeded CPU replay.
Request growth now reuses host table storage with suffix rollback. The benchmark
records admission failures, page waste, metadata allocations, retained host bytes,
and instrumented/aggregate timings. Chunk 07 adds an independent numerical oracle,
a checked eager PyTorch reference, and GPU preflight preparation. KV payloads,
CUDA normalization/attention kernels, and real serving remain planned. Recorded infrastructure spending is **$0**, with a **$100 ceiling** and
**$20 first spending checkpoint**.

Start with the [engineering roadmap](ROADMAP.md), [implementation log](docs/ENGINEERING_LOG.md), and code contracts below. Blog posts are local-only and are not distributed with the repository.

## Build and test

Install a C++17 compiler, CMake 3.24 or newer, and Ninja. On this ARM64 Mac the CPU implementation builds with Apple Clang; CUDA is not required. From this project directory:

```sh
cmake --preset debug
cmake --build --preset debug --parallel 4
ctest --preset debug
./build/debug/page_pool_demo
./build/debug/shared_prefix_demo
./build/debug/request_state_demo
```

Test builds require Python 3.9+ (standard library only) and fetch a SHA-256-pinned
GoogleTest archive. See [dependencies](docs/DEPENDENCIES.md) for offline setup and
builds with tests disabled. Separate `release`, `asan`, `ubsan`, and `asan-ubsan`
presets keep performance and diagnostic builds distinct. The legacy
`KVLAB_SANITIZE=ON` option still enables both sanitizers.

Chunk 07 passes 75 local Debug/Release checks and 76 UBSan checks. The optional
PyTorch CPU reference passes 10 unit tests and a 218-case shape/dtype matrix,
including a targeted test for rounded-residual semantics. Linux validation of
this chunk is pending at the implementation commit. See [the validation guide](docs/VALIDATION.md)
for executed results and the unresolved local Mac ASan initialization issue.
The [paired CPU allocator experiment](docs/experiments/0003-request-table-growth.md)
retains its 54 matching pairs, memory tradeoffs, and measured regression.

The original development directory ends in a space. Quote that path in shell commands; a fresh clone can use a conventional directory name.

Each completed implementation chunk has its own detailed commit. See the [chunk history and reproduction guide](docs/COMMIT_HISTORY.md) to inspect or build an earlier revision.

## Code and learning sequence

- [PagePool contract](include/kvlab/page_pool.hpp) and [implementation](src/allocator/page_pool.cpp): generations, checked counts, sealing, and physical-page accounting.
- [PageLease contract](include/kvlab/page_lease.hpp) and [implementation](src/allocator/page_lease.cpp): one owned reference per move-only lease; the pool must outlive it.
- [RequestState contract](include/kvlab/request_state.hpp) and [implementation](src/allocator/request_state.cpp): reserve capacity, declare committed tokens, share complete prefixes, and translate token addresses.
- [Request failure tests](tests/request_state_test.cpp) and [multi-request model](tests/request_model_test.cpp): deterministic failure injection and 64,000 request transitions, alongside the earlier 1,160,000 pool/sharing decisions per complete suite.
- [Page-pool demo](examples/page_pool_demo.cpp), [page-sharing demo](examples/shared_prefix_demo.cpp), and [request demo](examples/request_state_demo.cpp): runnable metadata examples.
- [Memory accounting](include/kvlab/memory_accounting.hpp): reconcile all owners, unique pages, reservation gaps, cache-only pages, and modeled KV bytes.
- [Workload replay](include/kvlab/workload.hpp) and [benchmark CLI](bench/allocator_bench.cpp): deterministic offered events, capacity failures, PMR counters, and reproducible output.
- [Benchmark contract](docs/BENCHMARKS.md) and [measured CPU report](docs/experiments/0002-cpu-allocator-screening.md): 108-run page-size/sharing sweep, raw-data provenance, timer limits, and charts.
- [Validation guide](docs/VALIDATION.md): independent sanitizer modes, fault-detection probes, compilation audits, and Linux CI.
- [Growth decision](docs/decisions/0007-request-table-growth.md) and [paired comparison](docs/experiments/0003-request-table-growth.md): suffix rollback, allocation reductions, retained-memory cost, and a measured regression.
- [Residual/RMSNorm contract](docs/RMSNORM.md), [independent oracle](python/kvlab/oracle.py), and [PyTorch reference](python/kvlab/reference.py): two outputs, explicit rounding, boundary checks, and 218 CPU correctness cases.
- [GPU preflight](docs/GPU_PREFLIGHT.md): bounded native/framework/profiler probes prepared for an existing GPU host; inventory and failure handling tested locally, device execution untested.
- Next: implement separate CUDA add/RMSNorm controls, then a fused kernel against the same contract. A GPU session must first close the actual compile/launch/debugger gates.

Request commitment declares materialized tokens; it does not write payloads or wait for a GPU. Borrowed request handles must not be manually retained, released, or sealed. The pool and any supplied metadata memory resource must outlive the request. CPU reference count zero does not prove GPU completion.

Before changing the implementation, trace the request demo and explain why reservation does not expose readable tokens, what rollback preserves, and why page generations can advance during a failed attempt. Each increment should remain small enough to review and explain personally.

## Working documents

- [Engineering log](docs/ENGINEERING_LOG.md): actual decisions, code changes, failures, diagnosis, validation, and lessons.
- [Initial architecture](docs/decisions/0001-scope-and-lifetime.md), [original pool contract](docs/decisions/0002-first-page-pool-contract.md), [sealed-page ownership](docs/decisions/0003-sealed-pages-and-leases.md), and [request transactions](docs/decisions/0004-request-transactions.md): alternatives and revisit conditions.
- [Budget ledger](docs/BUDGET.md) and [dated research](docs/RESEARCH_NOTES.md): cost gates and source observations.
- [Experiment template](docs/templates/EXPERIMENT.md): hypotheses, baselines, raw measurements, and conclusions.
- [Project guidance](AGENTS.md), [development conventions](CONTRIBUTING.md), and [team/subagent workflow](docs/TEAM_WORKFLOW.md): chunk boundaries, reviews, ownership, and continuous documentation.

The roadmap's repository tree includes proposed future files. Follow this README and the engineering log for implemented scope. CPU workload comparisons show regime-dependent changes; no GPU result or end-to-end serving capability is claimed.

## First CPU experiment

```sh
cmake --preset release
cmake --build --preset release --parallel 4
./build/release/allocator_bench --output artifacts/first-run --events 20000
```

The output directory must not already exist. Read the benchmark contract before
interpreting timings: zero-duration samples mean timer resolution limits, and
whole-trace means include skipped/rejected work. Modeled KV bytes are not device
memory allocations. The next GPU spending gate remains closed in this session.

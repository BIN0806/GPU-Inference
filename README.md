# LLM Kernel Inference

A C++/CUDA inference-components laboratory built in small, explained, testable increments: a fixed-page KV allocator, fused residual/RMSNorm, and single-layer paged decode attention.

**Implemented: chunks 01–03, the CPU allocator and request ownership layer.** The page pool has generation-tagged handles, checked references, and full-page sealing. Move-only leases balance ownership. `RequestState` adds per-request tables, separate reserved/committed lengths, committed-token address translation, transactional growth and prefix sharing, and cancellation cleanup. This is CPU metadata only: KV payloads, CUDA kernels, and performance benchmarks remain planned. Infrastructure spending is **$0**, with a **$100 ceiling** and **$20 first spending checkpoint**.

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

The first test-enabled configure fetches a SHA-256-pinned GoogleTest archive. See [dependencies](docs/DEPENDENCIES.md) for offline configuration and a tests-disabled build. Release and sanitizer configurations use the `release`, `ubsan`, and `asan` presets. All 52 checks passed without compiler warnings in clean Debug, Release, and UBSan-only builds for chunk 03. Combined ASan/UBSan startup remains unresolved from chunk 01 and was not rerun in this increment. Executed results and environment limitations belong in the validation record; a build configuration alone is not proof that a check passed.

The original development directory ends in a space. Quote that path in shell commands; a fresh clone can use a conventional directory name.

Each completed implementation chunk has its own detailed commit. See the [chunk history and reproduction guide](docs/COMMIT_HISTORY.md) to inspect or build an earlier revision.

## Code and learning sequence

- [PagePool contract](include/kvlab/page_pool.hpp) and [implementation](src/allocator/page_pool.cpp): generations, checked counts, sealing, and physical-page accounting.
- [PageLease contract](include/kvlab/page_lease.hpp) and [implementation](src/allocator/page_lease.cpp): one owned reference per move-only lease; the pool must outlive it.
- [RequestState contract](include/kvlab/request_state.hpp) and [implementation](src/allocator/request_state.cpp): reserve capacity, declare committed tokens, share complete prefixes, and translate token addresses.
- [Request failure tests](tests/request_state_test.cpp) and [multi-request model](tests/request_model_test.cpp): deterministic failure injection and 64,000 request transitions, alongside the earlier 1,160,000 pool/sharing decisions per complete suite.
- [Page-pool demo](examples/page_pool_demo.cpp), [page-sharing demo](examples/shared_prefix_demo.cpp), and [request demo](examples/request_state_demo.cpp): runnable metadata examples.
- Next chunk: CPU workload measurements, memory accounting, and fragmentation experiments before GPU spending.

Request commitment declares materialized tokens; it does not write payloads or wait for a GPU. Borrowed request handles must not be manually retained, released, or sealed. The pool and any supplied metadata memory resource must outlive the request. CPU reference count zero does not prove GPU completion.

Before changing the implementation, trace the request demo and explain why reservation does not expose readable tokens, what rollback preserves, and why page generations can advance during a failed attempt. Each increment should remain small enough to review and explain personally.

## Working documents

- [Engineering log](docs/ENGINEERING_LOG.md): actual decisions, code changes, failures, diagnosis, validation, and lessons.
- [Initial architecture](docs/decisions/0001-scope-and-lifetime.md), [original pool contract](docs/decisions/0002-first-page-pool-contract.md), [sealed-page ownership](docs/decisions/0003-sealed-pages-and-leases.md), and [request transactions](docs/decisions/0004-request-transactions.md): alternatives and revisit conditions.
- [Budget ledger](docs/BUDGET.md) and [dated research](docs/RESEARCH_NOTES.md): cost gates and source observations.
- [Experiment template](docs/templates/EXPERIMENT.md): hypotheses, baselines, raw measurements, and conclusions.
- [Project guidance](AGENTS.md), [development conventions](CONTRIBUTING.md), and [team/subagent workflow](docs/TEAM_WORKFLOW.md): chunk boundaries, reviews, ownership, and continuous documentation.

The roadmap's repository tree includes proposed future files. Follow this README and the blog index for implemented scope. No allocator speedup, GPU result, or end-to-end serving capability is claimed.

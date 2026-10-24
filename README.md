# LLM Kernel Inference

A C++/CUDA inference-components laboratory built in small, explained, testable increments: a fixed-page KV allocator, fused residual/RMSNorm, and single-layer paged decode attention.

**Implemented: chunks 01–02, the CPU page pool and shared-page ownership.** The pool provides fixed capacity, generation-tagged handles, checked reference counts, full-page sealing, physical-page statistics, and invariant diagnostics. Move-only `PageLease` objects manage references through scope exit, moves, and explicit sharing. Two demos and focused/model tests exercise these behaviors. This remains CPU metadata only: KV payloads, request tables, CUDA kernels, and performance benchmarks are still planned. Infrastructure spending is **$0**, with a **$100 ceiling** and **$20 first spending checkpoint**.

Start with the [engineering roadmap](ROADMAP.md), [implementation log](docs/ENGINEERING_LOG.md), and code contracts below. Blog posts are local-only and are not distributed with the repository.

## Build and test

Install a C++17 compiler, CMake 3.24 or newer, and Ninja. On this ARM64 Mac the CPU implementation builds with Apple Clang; CUDA is not required. From this project directory:

```sh
cmake --preset debug
cmake --build --preset debug --parallel 4
ctest --preset debug
./build/debug/page_pool_demo
./build/debug/shared_prefix_demo
```

The first test-enabled configure fetches a SHA-256-pinned GoogleTest archive. See [dependencies](docs/DEPENDENCIES.md) for offline configuration and a tests-disabled build. Release and sanitizer configurations use the `release`, `ubsan`, and `asan` presets. All 32 checks passed in clean Debug, Release, and UBSan-only builds for chunk 02. Combined ASan/UBSan startup remains unresolved from chunk 01 and was not rerun in this increment. Executed results and environment limitations belong in the validation record; a build configuration alone is not proof that a check passed.

The current directory name ends in a space. Quote its path in shell commands; do not inadvertently create a second project directory.

## Code and learning sequence

- [PagePool contract](include/kvlab/page_pool.hpp) and [implementation](src/allocator/page_pool.cpp): generations, checked counts, sealing, and physical-page accounting.
- [PageLease contract](include/kvlab/page_lease.hpp) and [implementation](src/allocator/page_lease.cpp): one owned reference per move-only lease; the pool must outlive it.
- [Pool tests](tests/page_pool_test.cpp), [sharing tests](tests/page_sharing_test.cpp), and [lease tests](tests/page_lease_test.cpp): 1,160,000 seeded decisions per complete run plus focused boundaries and ownership cases.
- [Page-pool demo](examples/page_pool_demo.cpp) and [shared-prefix demo](examples/shared_prefix_demo.cpp): runnable metadata examples.
- Next chunk: per-request page tables and transactional growth/sharing, including allocation-failure rollback.

Sealing uses a caller-declared filled-token count; payload/write validation comes later. Borrowed handles do not retain. Never manually release a lease's handle: reference counts cannot distinguish an owner's repeated release while other owners still exist. CPU reference count zero does not prove GPU completion.

Before changing the implementation, trace the shared-prefix demo and explain why a sealed page stays immutable at count one, why a move adds no reference, and which object must outlive the lease. Each increment should remain small enough to review and explain personally.

## Working documents

- [Engineering log](docs/ENGINEERING_LOG.md): actual decisions, code changes, failures, diagnosis, validation, and lessons.
- [Initial architecture](docs/decisions/0001-scope-and-lifetime.md), [original pool contract](docs/decisions/0002-first-page-pool-contract.md), and [sealed-page ownership](docs/decisions/0003-sealed-pages-and-leases.md): alternatives and revisit conditions.
- [Budget ledger](docs/BUDGET.md) and [dated research](docs/RESEARCH_NOTES.md): cost gates and source observations.
- [Experiment template](docs/templates/EXPERIMENT.md): hypotheses, baselines, raw measurements, and conclusions.
- [Project guidance](AGENTS.md), [development conventions](CONTRIBUTING.md), and [team/subagent workflow](docs/TEAM_WORKFLOW.md): chunk boundaries, reviews, ownership, and continuous documentation.

The roadmap's repository tree includes proposed future files. Follow this README and the blog index for implemented scope. No allocator speedup, GPU result, or end-to-end serving capability is claimed.

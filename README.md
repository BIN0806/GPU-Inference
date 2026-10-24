# LLM Kernel Inference

A C++/CUDA inference-components laboratory built in small, explained, testable increments: a fixed-page KV allocator, fused residual/RMSNorm, and single-layer paged decode attention.

**Implemented: chunk 01, the CPU page-ID pool.** It provides fixed capacity, exhaustion/reuse, generation-tagged handles, checked release, statistics, invariant diagnostics, a runnable demo, and GoogleTest checks. It owns CPU metadata only; KV payloads, request tables, sharing, CUDA kernels, and performance benchmarks are still planned. Infrastructure spending remains **$0**, with a **$100 ceiling** and **$20 first spending checkpoint**.

Start with the [engineering roadmap](ROADMAP.md), [implementation log](docs/ENGINEERING_LOG.md), and code contracts below. Blog posts are local-only and are not distributed with the repository.

## Build and test

Install a C++17 compiler, CMake 3.24 or newer, and Ninja. On this ARM64 Mac the CPU implementation builds with Apple Clang; CUDA is not required. From this project directory:

```sh
cmake --preset debug
cmake --build --preset debug --parallel 4
ctest --preset debug
./build/debug/page_pool_demo
```

The first test-enabled configure fetches a SHA-256-pinned GoogleTest archive. See [dependencies](docs/DEPENDENCIES.md) for offline configuration and a tests-disabled build. Release and sanitizer configurations use the `release` and `asan` presets. Debug, Release, and UBSan-only checks passed locally; combined ASan/UBSan startup remains unresolved on this host (including a minimal independent probe). Executed results and environment limitations belong in the validation record; a build configuration alone is not proof that a check passed.

The current directory name ends in a space. Quote its path in shell commands; do not inadvertently create a second project directory.

## Code and learning sequence

- [Public API](include/kvlab/page_pool.hpp): handle lifetime, failure results, and pool contract.
- [Implementation](src/allocator/page_pool.cpp): preallocated free-ID stack and generation checks.
- [Tests](tests/page_pool_test.cpp): edge cases and 160,000 seeded decisions against an independent set-based model.
- [Demo](examples/page_pool_demo.cpp): exhaust four pages, reclaim one, reject an old handle, and restore capacity.
- Next chunk: reference counts and explicit sharing of immutable complete pages; then per-request tables and transactional growth.

Before changing the implementation, explain why a live/free flag alone cannot reject a stale release after an ID is reused. Then trace the demo by hand. Each future increment should remain small enough to review and explain personally.

## Working documents

- [Engineering log](docs/ENGINEERING_LOG.md): actual decisions, code changes, failures, diagnosis, validation, and lessons.
- [Initial architecture](docs/decisions/0001-scope-and-lifetime.md) and [page-pool contract](docs/decisions/0002-first-page-pool-contract.md): alternatives and revisit conditions.
- [Budget ledger](docs/BUDGET.md) and [dated research](docs/RESEARCH_NOTES.md): cost gates and source observations.
- [Experiment template](docs/templates/EXPERIMENT.md): hypotheses, baselines, raw measurements, and conclusions.
- [Project guidance](AGENTS.md), [development conventions](CONTRIBUTING.md), and [team/subagent workflow](docs/TEAM_WORKFLOW.md): chunk boundaries, reviews, ownership, and continuous documentation.

The roadmap's repository tree includes proposed future files. Follow this README and the blog index for implemented scope. No allocator speedup, GPU result, or end-to-end serving capability is claimed.

# Documentation

The [project README](../README.md) introduces the goal, available capabilities,
limitations, and quick start. This directory holds component contracts,
reproduction details, and engineering records. Planned architecture is distinct
from implemented behavior and measured results.

## Setup and component guides

| Guide | Use it for |
|---|---|
| [Dependencies](DEPENDENCIES.md) | Compiler/build requirements, pinned dependencies, offline setup, and optional Python environments. |
| [Development guide](development/README.md) | Source-code map, allocator ownership rules, examples, and the implementation sequence. |
| [Residual + RMSNorm](RMSNORM.md) | Supported inputs, rounding semantics, reference setup, and numerical correctness results. |
| [GPU preflight](GPU_PREFLIGHT.md) | Toolchain and profiling checks on an existing GPU host, with inventory kept separate from successful execution. |
| [CPU benchmarks](BENCHMARKS.md) | Workload configuration, measurement boundaries, output schemas, and interpretation. |
| [Validation](VALIDATION.md) | Test/sanitizer modes, executed local and hosted checks, and known environment limitations. |

## Architecture and plans

- [Engineering roadmap](../ROADMAP.md): target architecture, prerequisites, phases, experiments, and decision gates. It is a design baseline, not evidence of completed work.
- [System scope and lifetime](decisions/0001-scope-and-lifetime.md): CPU/GPU responsibilities and the intended system boundary.
- Allocator decisions: [page-pool contract](decisions/0002-first-page-pool-contract.md), [sealed pages and leases](decisions/0003-sealed-pages-and-leases.md), [request transactions](decisions/0004-request-transactions.md), and [request-table growth](decisions/0007-request-table-growth.md).
- Numerical and validation decisions: [RMSNorm contract](decisions/0008-rmsnorm-contract.md), [CPU measurement](decisions/0005-cpu-workload-measurement.md), and [sanitizer validation](decisions/0006-sanitizer-validation.md).

## Experiments and evidence

- [CPU allocator screening](experiments/0002-cpu-allocator-screening.md): page-size and sharing experiments, raw-data provenance, and measurement limits.
- [Request-table growth comparison](experiments/0003-request-table-growth.md): paired allocation, retained-memory, and timing results, including a regression.
- [RMSNorm CPU case records](validation/0007-rmsnorm-cpu.json) and [Linux validation summary](validation/0007-linux-ci.json): numerical and execution evidence.
- [Experiment template](templates/EXPERIMENT.md): record hypotheses, configurations, baselines, measurements, and conclusions.

## Development records

- [Implementation history](COMMIT_HISTORY.md): chunk-by-chunk commits and revision reproduction.
- [Engineering log](ENGINEERING_LOG.md): design choices, code changes, encountered problems, diagnosis, and executed checks.
- [Budget ledger](BUDGET.md): infrastructure spending, resource cleanup, and spending checkpoints.
- [Research notes](RESEARCH_NOTES.md): dated external sources and infrastructure observations.
- [Team workflow](TEAM_WORKFLOW.md) and [contributing conventions](../CONTRIBUTING.md): ownership, review, integration, and documentation practices.

Keep changing test counts, run links, spending totals, and implementation history
in these records. The main README summarizes the project without reproducing its
development journal.

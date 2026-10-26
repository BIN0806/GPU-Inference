# LLM Kernel Inference

A C++ and CUDA systems project exploring how memory management, kernel fusion,
and paged attention affect LLM inference. The goal is a small, measurable
inference-components laboratory with explicit correctness contracts and
reproducible performance experiments.

The target system combines a C++ KV-cache page allocator with GPU-resident KV
storage and single-layer paged decoding. A parallel kernel workstream studies
fused residual addition and RMSNorm. Together, these components provide a way to
investigate memory utilization, bandwidth, and decode latency.

## What is available

| Component | Current capabilities |
|---|---|
| C++ KV-cache allocator | Fixed-size page management, per-request page tables, reference-counted sharing of immutable prefixes, and memory accounting. Currently manages metadata rather than KV tensor payloads. |
| Synthetic CPU workloads | Reproducible request replay and benchmarks for allocation behavior, page utilization, and host-side latency. |
| Residual + RMSNorm references | An independent numerical oracle and an eager PyTorch implementation with an explicit rounding contract and correctness checks. Validated on CPU. |
| Development tooling | CMake builds, unit and model tests, host sanitizer checks, Linux CI, and GPU environment preflight scripts. GPU execution probes are prepared but not yet validated on hardware. |

## What is not implemented yet

- GPU KV-cache storage and allocator-to-device integration.
- CUDA/Triton normalization kernels and paged decode-attention kernels.
- GPU completion-aware memory reclamation and integrated decoding benchmarks.

This project does not currently load models, generate text, or provide a serving
API. Its core target is a **single-layer synthetic inference benchmark**; a
complete model-serving engine is outside the core scope. GPU performance results
are not yet available.

## Quick start

The CPU components require a C++17 compiler, CMake 3.24+, Ninja, and Python 3.9+.
They have been validated on macOS ARM64 and Linux x86-64. CUDA and PyTorch are
not required for the C++ build or its tests.

```sh
git clone https://github.com/BIN0806/GPU-Inference.git
cd GPU-Inference
cmake --preset debug
cmake --build --preset debug --parallel 4
ctest --preset debug
```

The first test-enabled configure downloads a checksum-pinned GoogleTest archive.
See [dependency setup](docs/DEPENDENCIES.md) for offline builds and optional
Python environments.

Run the request-lifecycle example after building:

```sh
./build/debug/request_state_demo
```

For the optional PyTorch reference, follow the platform-specific setup in
[the RMSNorm guide](docs/RMSNORM.md). To run a CPU workload experiment, use
[the benchmark guide](docs/BENCHMARKS.md). Supported validation modes and known
platform limitations are documented in [the validation guide](docs/VALIDATION.md).

## Documentation and contributing

Start with the [documentation index](docs/README.md) for component guides,
architecture decisions, experiment results, and development records. The
[roadmap](ROADMAP.md) describes the intended architecture and future work.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development and review conventions.

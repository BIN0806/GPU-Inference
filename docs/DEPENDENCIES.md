# Dependencies for the CPU implementation (through chunk 07)

| Component | Requirement / pin | Purpose |
|---|---|---|
| C++ compiler | C++17; local validation uses Apple Clang 17 | CPU allocator and demo |
| CMake | 3.24 or newer | Build targets and presets |
| Ninja | Required by supplied presets | Build execution |
| Python | 3.9+; standard library only | Required for test-enabled builds; no production runtime dependency |
| GoogleTest | v1.17.0 archive, SHA-256 below | Test-only; fetched only when BUILD_TESTING is enabled |

GoogleTest source: [official v1.17.0 release](https://github.com/google/googletest/releases/tag/v1.17.0).
Archive URL: `https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0`.

```text
SHA256 65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
```

The digest was computed from the official downloaded archive and is enforced by CMake for both network and optional local-archive paths. The release is deliberately fixed; it is not advertised as latest. GoogleTest's BSD 3-Clause license is retained in its fetched source. No third-party source is vendored into the project.

To configure offline after obtaining that exact archive:

```sh
cmake --preset debug -DKVLAB_GTEST_ARCHIVE=/absolute/path/to/archive.tar.gz
```

To build just the library and demo without fetching test dependencies:

```sh
cmake -S . -B build/library-only -G Ninja -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/library-only
./build/library-only/page_pool_demo
```

CUDA, PyTorch, Triton, and Python packages are not runtime or build dependencies of chunk 01. At chunk 01, no CI service had been run; current validation is recorded in [the validation guide](VALIDATION.md).

On the initial macOS host, combined ASan/UBSan compiled but hung during startup, including for an independent minimal probe. Debug, Release, and UBSan-only checks passed. See the validation record (local-only record) for reproduction and the unresolved ASan item.

Chunk 02 adds no third-party dependencies. The `ubsan` configure/build/test presets capture the previously executed UBSan-only flags for Clang/GCC on macOS/Linux. At chunk 02, `asan` enabled both address and undefined sanitizers; chunk 05 changes that preset to ASan only and adds `asan-ubsan`. All 32 chunk 02 checks passed in Debug, Release, and UBSan-only builds; see part 02 evidence (local-only record).

Chunk 03 uses standard C++17 `<memory_resource>` for request table allocation and isolated failure injection. No third-party dependency was added. The pool and selected memory resource must outlive their requests. See part 03 validation (local-only record) for the 52-check matrix and remaining limitations.

## Chunk 04 tools

The workload and accounting libraries use C++17 only; no production dependency
was added. Python 3 (standard library) enables `BenchmarkOutputContract` in CTest
and runs the serial sweep. The CMake check was optional in chunk 04. In chunk 05, test-enabled builds
require Python so missing tooling cannot silently reduce coverage. Tested local
Python: 3.14.8.

Plotting is optional and isolated from compilation and measurement. Install the
pinned package set in `scripts/requirements-plot.txt` into a virtual environment.
The lock records the tested macOS ARM64/Python 3.14 environment; other platforms
may require another reviewed lock. Matplotlib 3.10.8 creates the static charts.
Its distribution and those of its dependencies include their own license files.
The experiment provenance records the versions used for the checked figures.
Neither plotting nor dependency installation provisions a GPU.

## Chunk 05 validation infrastructure

No C++ runtime library was added. Python 3.9+ is required only with
`BUILD_TESTING=ON`. Test scripts use the standard library. The
[CI workflow](../.github/workflows/cpu-ci.yml) uses standard `ubuntu-24.04`
GitHub-hosted runners and one external action:

- [actions/checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1),
  pinned to `3d3c42e5aac5ba805825da76410c181273ba90b1` (MIT license, Node 24 runtime).
- GoogleTest remains the archive above, verified on every fresh configure.
- The runner image and installed compiler/CMake/Ninja/Python are recorded in job
  logs. `ubuntu-24.04` is a moving image, not a fully pinned container. A changed
  image is a reproduction variable; record it before attributing a regression
  to source changes. The five CMake jobs do not install compiler or Python packages. The separate
  chunk 07 reference job installs the pinned CPU Python dependencies described below.

The action tag and commit were checked on 2026-10-25. Updating it is a reviewed
source change. Checkout does not persist credentials; jobs have read-only
repository access and receive no project secrets. See [ADR 0006](decisions/0006-sanitizer-validation.md)
and [the budget ledger](BUDGET.md) for the rationale and cost boundary.

Chunk 06 adds no dependency. The paired comparison uses Python's standard library;
its optional plotting script uses the existing Matplotlib lock. Both measured
binaries were built with Apple Clang 17 / libc++ 200100 on an Apple M3 Pro. Compiler
and source provenance are linked from [EXP-0003](experiments/0003-request-table-growth.md).

## Chunk 07 optional PyTorch reference

The C++ library, standard CMake suite, FP64 oracle, and preflight orchestration
still need no third-party Python package. Running the tensor reference adds an
isolated Python 3.12 environment with PyTorch 2.14.1. Two complete hash-checked
wheel locks keep that dependency outside the allocator build:

- [macOS ARM64 lock](../scripts/requirements-rmsnorm-macos.txt): CPython 3.12,
  macOS 14+, PyPI wheels; installed and executed on the local M3 Pro.
- [Linux x86-64 CPU lock](../scripts/requirements-rmsnorm-linux-cpu.txt): CPython
  3.12, official PyTorch CPU index. Exact wheel bytes were downloaded and hashed
  locally; Linux execution is pending at the implementation commit.

Install with `python -m pip install --require-hashes -r <lock>` inside a venv,
then `python -m pip check`. Each lock fixes all resolved dependencies, not only
Torch. Different platform indexes resolved different transitive versions; do
not merge the two environments. PyTorch carries a BSD-style license and bundled
third-party notices; dependency licenses remain in their distributions. No
wheel or third-party source is committed.

The local installation passes `pip check`. NumPy is intentionally absent: no
reference code uses its bridge. The observed startup warning is retained rather
than suppressed. CPython 3.14.0's local certificate setup rejected one metadata
request; a TLS-verifying curl request succeeded, and the Homebrew Python 3.12
venv installed the wheels without weakening certificate verification.

These are **CPU correctness locks**, not a CUDA development-image specification.
Do not install them over a rented GPU environment. The remote driver, toolkit,
PyTorch CUDA wheel, Triton, and profiler tuple still needs selection and actual
execution. [Reference setup](RMSNORM.md), [preflight](GPU_PREFLIGHT.md),
[official PyTorch package metadata](https://pypi.org/pypi/torch/json).

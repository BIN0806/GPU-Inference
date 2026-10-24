# Dependencies for the CPU implementation (chunks 01–02)

| Component | Requirement / pin | Purpose |
|---|---|---|
| C++ compiler | C++17; local validation uses Apple Clang 17 | CPU allocator and demo |
| CMake | 3.24 or newer | Build targets and presets |
| Ninja | Required by supplied presets | Build execution |
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

CUDA, PyTorch, Triton, and Python packages are not runtime or build dependencies of chunk 01. No CI service has been run; Linux portability checks and CI configuration are subsequent work.

On the initial macOS host, combined ASan/UBSan compiled but hung during startup, including for an independent minimal probe. Debug, Release, and UBSan-only checks passed. See the validation record (local-only record) for reproduction and the unresolved ASan item.

Chunk 02 adds no third-party dependencies. The `ubsan` configure/build/test presets capture the previously executed UBSan-only flags for Clang/GCC on macOS/Linux. `asan` still enables both address and undefined sanitizers. All 32 chunk 02 checks passed in Debug, Release, and UBSan-only builds; see part 02 evidence (local-only record).

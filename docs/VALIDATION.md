# CPU validation: flags, runtime, then behavior

Chunk 05 separates three questions: did the compiler instrument our code, can
that runtime detect a known fault on this host, and does the actual allocator
pass its suite? All three need evidence. A successful build answers only the
first question partially; a generic crash is not a passing fault-detection test.

## Build modes and migration

| Preset | Configuration | Expected sanitizer set | Registered checks |
|---|---|---|---:|
| `debug` | Debug | none | 69 |
| `release` | Release | none | 69 |
| `asan` | Debug | address | 70 |
| `ubsan` | Debug | undefined | 70 |
| `asan-ubsan` | Debug | address, undefined | 70 |

`KVLAB_ENABLE_ASAN` and `KVLAB_ENABLE_UBSAN` independently select instrumentation.
`KVLAB_SANITIZE=ON` remains a legacy alias for both. All presets explicitly clear
that alias; an explicit `-D` override still wins. The `asan` preset previously
selected both sanitizers. Select `asan-ubsan` when reproducing that combination.

The old UBSan preset placed flags in `CMAKE_CXX_FLAGS`. **Use a fresh build tree**
when migrating: `cmake --fresh --preset ubsan` clears the old cache. Export any
local cache settings you need first. This project does not silently erase user
compiler flags. The flag audit rejects unexpected instrumentation and disabling
overrides. Do not reuse a sanitizer binary for timing experiments.

Python 3.9+ is required for test-enabled builds. A tests-disabled library/benchmark
build does not require Python or GoogleTest. This deliberately replaces the old
optional Python check, which could silently reduce the executed suite.

```sh
cmake --preset ubsan
cmake --build --preset ubsan --parallel 2
ctest --preset ubsan --timeout 120
```

For an offline configure, append
`-DKVLAB_GTEST_ARCHIVE=/absolute/path/to/the/pinned/archive.tar.gz`.
Use the [dependency checksum](DEPENDENCIES.md). Do not download an arbitrary
archive and bypass the digest.

## Isolate runtime trouble before GoogleTest discovery

```sh
cmake --preset asan
cmake --build --preset asan --parallel 2
python3 scripts/check_sanitizer_flags.py \
  --source-dir . --build-dir build/asan --expected address
python3 tests/sanitizer_runtime_test.py build/asan/sanitizer_probe \
  --sanitizers address --log-dir artifacts/asan-preflight
```

The audit independently enumerates first-party C++ sources and checks their
`compile_commands.json` entries, including frame-pointer flags. Missing source
entries, missing/extra sanitizer sets, and disabling overrides fail. It does not
claim to emulate every compiler option, inspect machine code, or instrument
GoogleTest internals. Compiler/linker flags are attached to the project's
interface target; the probe's actual execution also checks that the linked
runtime can operate.

The probe first runs a valid allocation and requires normal exit plus a marker.
It then deliberately writes one element beyond a four-element heap array for
ASan, or overflows a signed integer for UBSan. The parent requires nonzero exit
**and the matching sanitizer diagnostic**. It rejects a timeout, segmentation
fault without that diagnostic, and a recovered fault that exits successfully.
Each child has a 20-second limit; the enclosing CTest has a 70-second limit.
Raw stdout, stderr, command, return code, and timeout status are written to JSON.
The probe is a separate test-only executable, never linked into the allocator.

`ValidationTools` regression cases check false-green failure paths without
executing undefined behavior. `SanitizerRuntime` is registered only in sanitized
builds. The unchanged allocator model suites still exercise 1,264,000 decisions
per full run. Counts include individual GoogleTests, demos, and Python checks;
69/70 are not counts of independent statistical trials.

Sanitizer success is evidence about these executed paths and toolchains, not a
proof that all memory bugs or undefined behavior are absent. There is no CUDA,
thread-race, or GPU-lifetime coverage in this matrix. Sanitizer timings are not
performance measurements. [ASan documentation](https://clang.llvm.org/docs/AddressSanitizer.html),
[UBSan documentation](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html).

## Hosted workflow and trust boundary

[CPU validation](../.github/workflows/cpu-ci.yml) runs all five presets on a
standard Ubuntu 24.04 GitHub-hosted runner for relevant pushes to `main` and
`codex/**`, pull requests, or manual dispatch. It records the image and tool
versions, builds, checks instrumentation/probes before discovery, then runs the
entire CPU suite. Jobs use GCC as provided by the runner and bounded timeouts.
Linux ASan jobs explicitly enable leak detection. The workflow calls CTest on
the build tree so test-preset environment variables cannot replace that setting.
Local presets leave platform-specific leak defaults alone.

Checkout is pinned to an immutable commit, does not retain credentials, and
receives read-only contents access. There are no secrets, paid runner labels,
cache steps, or artifact uploads. A private-repository guard prevents automatic
execution after a visibility change. See [cost rules](BUDGET.md). Ordinary
`pull_request` is used; this does not run untrusted changes in a privileged
`pull_request_target` context. Logs are public, so never print credentials.

Branch protection and required checks have **not** been configured: this is
executed CI, not a claim that GitHub will prohibit every bad merge. The runner
image can change. A failure should be reproduced using the recorded compiler,
image, and source SHA before changing allocator code. Documentation-only changes
do not trigger automatic jobs; use manual dispatch for a new verification run.

## Executed evidence — 2026-10-25

Local host: Apple Clang 17, ARM64 macOS 26.6.2, CMake 4.4.4, Ninja 1.13.2,
Python 3.14.8 selected by CMake (direct shell scripts used Python 3.14.0). Fresh Debug and Release configurations pass all 69 checks; fresh
UBSan passes all 70. All five modes compile, and their first-party flag audits
pass. ASan-only and combined clean probes time out after 20 seconds on this
host, including after adding a flushed entry-to-main stderr marker; neither
mode reached that marker. The refined UBSan runtime, flag audit, and validation
tool regressions passed again (3/3). The legacy combined option built its probe
and passed its compile audit; a tests-disabled Release build ran benchmark help.
No full ASan suite is claimed locally. The earlier startup symptom is
therefore still an open environment issue, not a diagnosed allocator failure.

Local raw build/test/probe records are retained under the ignored
`artifacts/validation/chunk-05/` directory. Hosted execution is pending at the
implementation commit; a follow-up evidence record will link the actual run,
source revision, toolchain, and results. No GPU resource or paid storage was
provisioned. No performance experiment was repeated for this build-only change.

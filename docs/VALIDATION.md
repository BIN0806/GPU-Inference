# CPU validation: flags, runtime, then behavior

Chunk 05 separates three questions: did the compiler instrument our code, can
that runtime detect a known fault on this host, and does the actual allocator
pass its suite? All three need evidence. A successful build answers only the
first question partially; a generic crash is not a passing fault-detection test.

## Build modes and migration

| Preset | Configuration | Expected sanitizer set | Registered checks |
|---|---|---|---:|
| `debug` | Debug | none | 82 |
| `release` | Release | none | 82 |
| `asan` | Debug | address | 83 |
| `ubsan` | Debug | undefined | 83 |
| `asan-ubsan` | Debug | address, undefined | 83 |

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
these totals are not counts of independent statistical trials.

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
`artifacts/validation/chunk-05/` directory. Hosted execution completed successfully in
[run 40172396157](https://github.com/BIN0806/GPU-Inference/actions/runs/40172396157) at source
[`a5041ec`](https://github.com/BIN0806/GPU-Inference/commit/a5041ece6e01c559161c354474239bad22ab4928). No GPU resource or paid storage was
provisioned. No performance experiment was repeated for this build-only change.

### Completed Linux matrix

| Mode | Checks | Runtime preflight | Outcome |
|---|---:|---|---|
| Debug | 69/69 | Not instrumented | Pass |
| Release | 69/69 | Not instrumented | Pass |
| ASan | 70/70 | Clean + heap overflow | Pass |
| UBSan | 70/70 | Clean + signed overflow | Pass |
| ASan + UBSan | 70/70 | Clean + both faults | Pass |

All jobs used x86-64 Ubuntu 24.04 image `20261023.334.3`, GCC
`13.3.0-6ubuntu2~24.04.1`, CMake 3.31.6, Ninja 1.13.2, and Python 3.12.3.
The run completed on 2026-10-25 at 07:22 UTC (00:22 America/Los_Angeles).
The compile audit checked 16 project translation units in normal builds and 17
in sanitized builds. All expected probe signatures and exit behaviors passed;
no compiler warnings or unexpected sanitizer reports appeared in the saved job
logs. ASan leak detection was enabled for the suite. All five jobs completed,
and the run's artifact listing returned zero uploads. Raw logs and job metadata
were downloaded locally under the ignored validation directory.

This closes the missing Linux ASan execution gate for the current CPU source.
It does not resolve the Mac's pre-main failure, add device coverage, or claim a
performance improvement. The follow-up documentation commit changes no code,
tests, build configuration, or workflow relative to the tested source above.

## Chunk 06 validation

The four new request-growth tests increase the current matrix to 73 normal and
74 sanitized checks. Debug/Release/UBSan passed locally; the updated comparison
harness's negative-path checks also passed. Existing model tests remain unchanged.
The historical chunk 05 runs above retain their original 69/70 counts. Chunk 06
[Linux run 40232916754](https://github.com/BIN0806/GPU-Inference/actions/runs/40232916754) completed successfully at
[`d0ad3a8`](https://github.com/BIN0806/GPU-Inference/commit/d0ad3a8b459fa52ce1cfa16964a7b783b485eafb): 73/73 in
Debug and Release, 74/74 in ASan, UBSan, and combined. All preflight probes and
flag audits passed. GCC 13.3.0, Ubuntu 24.04 image `20261023.334.3`, CMake 3.31.6,
Ninja 1.13.2, Python 3.12.3. No compiler warnings or unexpected sanitizer reports
appeared in the retained logs. All jobs completed; artifact count was zero. Performance evidence is
kept separate in [EXP-0003](experiments/0003-request-table-growth.md).

## Chunk 07 reference and preflight validation

Two new CTest entries (`RmsNormOracle` and `GpuPreflightContracts`) increase the
current total to **75 normal / 76 sanitized** checks. Fresh local Debug and
Release builds pass 75/75; fresh UBSan passes 76/76. After adding a final
preflight deadline regression, the two affected CTest entries passed again.
The current preflight entry includes six unittest methods covering classification,
real missing-program/timeout behavior, failed compilation, exhausted global
budget, non-overwrite behavior, and failed framework/profiler fixtures. Fixture
strings are not device evidence. No allocator implementation changed.

The separate optional PyTorch CPU environment passes 10 unit tests plus all
218 shape/dtype cases, with bitwise residual equality and zero normalized-output
threshold failures. [Numerical contract and results](RMSNORM.md) explain the
error metrics and exact rounding witness. The complete local case records and
source hashes are in [the result JSON](validation/0007-rmsnorm-cpu.json).
Local toolchain remains Apple Clang 17, macOS ARM64, CMake 4.4.4, Ninja 1.13.2;
reference Python 3.12 / Torch 2.14.1. Raw logs are retained locally in
`artifacts/validation/chunk-07/`. No RMSNorm performance result is claimed.

The workflow now has six jobs: the existing five CMake configurations and a
separate hash-pinned Linux CPU PyTorch reference job. The latter executes the
same 10 unit tests and 218-case matrix. Hosted execution was pending at the implementation commit; the completed run
below establishes this source independently of the earlier dated runs.

The actual local GPU inventory finds no CUDA/profiler tools; its report correctly
keeps execution and counters untested. The prepared CUDA smoke source and
Triton/compile/device/profiler paths have not run on hardware. The previously
observed local ASan startup limitation remains open; no local ASan pass is added.

### Completed chunk 07 Linux evidence

[run 40270856228](https://github.com/BIN0806/GPU-Inference/actions/runs/40270856228) completed successfully at source
[`e76e02b`](https://github.com/BIN0806/GPU-Inference/commit/e76e02b602eacfca184e1a9dff0fb7c9a25d8061)
on 2026-10-26 04:49 UTC (2026-10-25 21:49 America/Los_Angeles).

| Job | Executed checks | Result |
|---|---|---|
| Debug / Release | 75 each | Pass |
| ASan / UBSan / combined | 76 each, including runtime probes | Pass |
| PyTorch CPU reference | 10 unit tests and 218 matrix cases | Pass |

The runner was Ubuntu 24.04 image `20261023.334.3`, GCC 13.3.0, CMake 3.31.6,
Ninja 1.13.2, Python 3.12.3, and Torch 2.14.1+cpu. Hash-checked Linux dependency
installation and `pip check` passed. No C++ compiler warnings or unexpected
sanitizer reports appear in the saved logs. The known optional NumPy warning
appears in the reference job and is retained. All residual bits match; output
error maxima match the local run, with a small FP32 normalized RMS difference
that remains below the unchanged threshold. See [the numerical summary](validation/0007-linux-ci.json).

All six jobs completed and the artifact listing returned zero uploads. Combined
logs, run/job metadata, and the empty artifact listing are retained locally;
the compact public summary records the combined log hash. The Linux per-case
JSON was generated but not uploaded, so only its printed aggregate metrics are
retained from that runner. Local per-case data remains available above. This
follow-up changes documentation/evidence only; executable source and workflow
are identical to the tested implementation commit. No GPU result is inferred.

## Chunk 08 native CUDA preparation

Six new C++ descriptor tests bring the current CTest totals to **81 normal /
82 sanitized**. Fresh local Debug/Release pass 81 each and UBSan passes 82,
including the first-party source audit. No compiler warnings appeared. These
checks exercise shape/type rejection, empty no-ops, byte capacities, alignment,
all-buffer overlap, address overflow, and wide element counts without touching
a GPU. Local ASan remains the earlier unresolved host environment limitation.

The seventh CI job compiles/links native CUDA for sm_86 and sm_89 using pinned
NVIDIA archives on a standard CPU runner. Its `--host-only` checker validates
linked empty/invalid paths and independent oracle witnesses with no driver.
Those checks were pending at the implementation commit; the completed result follows below. GPU smoke/full matrices
and Compute Sanitizer are prepared but unexecuted. Read [the CUDA guide](CUDA_RMSNORM.md)
for the difference between host, compiler, and hardware evidence. No kernel
performance measurement is reported.

### Completed chunk 08 Linux evidence

[run 40427805616](https://github.com/BIN0806/GPU-Inference/actions/runs/40427805616) passed at source `d0d21f2` on 2026-10-26 America/Los_Angeles.
All five CMake configurations passed (81 normal / 82 sanitized), the Python
reference passed 10 unit tests and 218 CPU cases, and the native CUDA job
compiled/linked sm_86/sm_89 then passed its no-GPU host checks. Compiler/resource
and environment details are in [the retained summary](validation/0008-linux-ci.json).
No C++/CUDA compiler warnings or unexpected sanitizer reports appear in the
saved logs. The existing optional NumPy warning remains in the reference job.

The first run 40427429877 compiled/linked but failed loading `libcudart.so.12`
before main. A separate fix commit selects the shared runtime consistently and
sets an explicit build RUNPATH from the imported CUDA target. The successful
run's `readelf` and `ldd` output shows the expected private-prefix runtime path,
then `{"host_checks":"passed","device_execution":false}`. Initial failure
and final success logs are retained locally; their digests appear in the summary.

No GPU smoke/full cases, Compute Sanitizer, or performance tests were executed.
The local configuration also rejected missing CUDA architecture settings, and
the Linux-only toolchain installer rejected the Mac without creating its target
directory. Seven completed hosted jobs uploaded zero artifacts. This evidence
follow-up changes docs only; it does not rerun or alter kernel arithmetic.


## Chunk 09 native session preparation

A new CTest group runs twelve native-session contract tests; the existing
preflight group adds a seventh interruption regression. Current totals are
**82 normal / 83 sanitized**. Local Debug passed 82 checks. After the final
parser/regression additions, both affected CTest groups and their individual
12/7-test suites passed again. Local ASan was not retried; its documented
startup issue remains unchanged.

Tests cover complete/unique case sets, malformed/nonfinite output, failed or
interrupted processes, missing/nonzero sanitizer summaries (including race
warnings), fail-fast sequencing, new-output-directory enforcement, source/binary
changes, and overall deadline exhaustion. Real subprocess tests exercise
process-group termination on timeout, SIGINT, and SIGTERM, including children
holding inherited output pipes. GPU success records are synthetic fixtures;
they are not retained as hardware evidence. Logs are local under
`artifacts/validation/chunk-09/`.

An actual Mac host-only attempt recorded missing nvcc, failed CMake configuration,
exit 2, and `hardware_checks_passed=false`; no build or device command followed.
This verifies the absence of a toolkit is a failed gate, not a successful skip.
The Linux workflow now drives the fresh sm_86 compile/host path through the new
runner and retains sm_89 compile/host checks. CI is pending at the implementation
commit. No CUDA arithmetic or tolerance changed, and no GPU result is claimed.


### Completed chunk 09 Linux evidence

[Run 40469386543](https://github.com/BIN0806/GPU-Inference/actions/runs/40469386543) completed at source `a72a702` on
2026-10-27 07:50:39 UTC (00:50:39 America/Los_Angeles). All seven jobs passed:
Debug/Release 82 each; ASan/UBSan/combined 83 each; CPU reference 10 unit tests
and 218 cases; native sm_86 fresh-build runner plus sm_89 compile/host checks.
The three sanitizer configurations include their expected-fault runtime probes
and first-party instrumentation audits. No code changes were needed after CI.

The host runner reported `host_verified` with `hardware_checks_passed=false`,
complete configure/build/host/hash gates, and an empty tracked Git status. Its
retained source hashes match the tested commit. The binary fingerprint and
compact environment/job record are in [the summary](validation/0009-linux-ci.json).
The raw host report was extracted from the completed job log and saved locally,
along with combined logs, metadata, and their hashes. Shared-runtime loader
checks passed. No C++/CUDA compiler warnings or unexpected sanitizer diagnostics
appeared; the existing optional NumPy warning remains in the reference job.

All jobs completed with zero artifact uploads. This follow-up updates evidence
and documentation only; no executable source, test, workflow, or tolerance
changed. Actual GPU execution remains the next gate.

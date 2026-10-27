# Native CUDA validation session

The [runner](../scripts/check_rmsnorm_cuda.py) prepares and checks the existing
unfused residual/RMSNorm control on an **already provisioned** Linux CUDA host.
It creates a fresh build, runs correctness gates, and retains diagnostics.
It performs no provisioning, package installation, timing sweep, permission
changes, or provider cleanup. Native validation does not require PyTorch or
Triton. Their separate readiness checks remain in [GPU preflight](GPU_PREFLIGHT.md).

## Prepare before paid time

1. Use a clean checkout of the intended commit and read [the native contract](CUDA_RMSNORM.md).
2. Confirm the selected development image includes CMake 3.24+, Ninja, Python
   3.9+, CUDA Toolkit 12.8+, a compatible host C++ compiler, and Compute
   Sanitizer. A runtime-only image is insufficient. Record its immutable image
   digest separately; the runner cannot discover the provider's image identity.
3. Record the actual device architecture and driver, fresh all-in quote,
   storage/transfer charges, maximum session time, and cleanup steps in
   [the budget ledger](BUDGET.md). The execution deadline below is not a billing
   limit and does not include provisioning, downloads, or artifact export.
4. Resolve build/dependency issues before renting when possible. The CPU CI job
   exercises the same fresh-build runner in explicit host-only mode. CUDA
   compilation and linked host checks do not establish device correctness.

## Execute one bounded native session

```sh
# 86 is an example; substitute the actual device's numeric compute capability.
python3 scripts/check_rmsnorm_cuda.py --run-gpu --cuda-arch 86 \
  --output artifacts/native-gpu-001 --max-seconds 600
```

Run from the repository root. The output directory must not exist. If nvcc is
outside PATH, add `--cuda-compiler /absolute/path/to/nvcc`. CUDA architecture
selection is mandatory and accepts one numeric target. This runner deliberately
does not inherit an old CMake cache or execute a supplied stale binary.

The ordered gates are:

1. Record source hashes, Git revision/tracked status and selected tool versions;
   configure Release with CUDA enabled and testing disabled; build only the
   native checker with two parallel jobs. GoogleTest is not downloaded.
2. Run linked host-only checks. Then, in GPU mode, inventory the GPU/driver and
   Compute Sanitizer and run all **34 smoke cases**.
3. Run the same smoke checker under **memcheck**, **racecheck**, and **synccheck**,
   stopping at the first failure. Each invocation needs both complete numerical
   output and the expected clean tool summary. Warnings/hazards fail racecheck.
4. Run the **218-case full matrix** without instrumentation; confirm relevant
   source files and the built binary still match their recorded hashes.

Every command has a timeout capped by the remaining overall deadline. A failed
required check stops subsequent gates. Missing tools, unknown report formats,
nonzero returns, timeouts, and interrupted commands cannot pass. Inventory
commands are diagnostic; actual build/execution/tool gates establish readiness.
The runner checks numerical output identities, uniqueness, finite nonnegative
error metrics, residual equality flags, valid error indices, and terminal case
counts. Numerical tolerance enforcement itself belongs to the independent
native checker; parsing JSON is not another numerical oracle.

Compute Sanitizer output goes to separate per-tool logs through `--log-file`,
so it cannot be confused with the checker's JSON lines. The runner requests
`--error-exitcode 99` and informational report visibility. It requires exactly
one clean summary; unsupported text formats remain failures until inspected.
Memcheck precedes racecheck because racecheck does not check memory-access
validity. [NVIDIA Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

## Read the evidence, not just the process return code

`validation.json` is replaced atomically after each command and at completion.
It includes argv, stdout/stderr, exit/status, orchestration elapsed time, source
and binary SHA-256 hashes, environment inventory, active/last command, individual
checks, and start/end timestamps. Raw sanitizer logs and the build tree remain
beside it. Tool logs are also embedded in the report for a compact export.
There is no full environment dump, credentials, or other users' process listing.
Review logs for incidental host details before publishing them.

| Result | Exit | Meaning |
|---|---:|---|
| `host_verified`, `hardware_checks_passed=false` | 0 | Fresh build and linked host checks only. |
| `hardware_verified`, `hardware_checks_passed=true` | 0 | All native numerical and sanitizer gates completed on this host. |
| `failed_*`, hardware flag false | 2 | Inspect the named check and its command/log. Later gates did not run. |
| `interrupted`, hardware flag false | 130 | Ctrl-C or SIGTERM stopped the session; partial output retained. |
| `running` or absent/incomplete report | Not a pass | The run never recorded successful completion. |

Host-only invocation for a compiler-equipped CPU machine:

```sh
python3 scripts/check_rmsnorm_cuda.py --host-only --cuda-arch 86 \
  --output artifacts/native-host-001 --max-seconds 300
```

Neither host mode nor unit-test fixtures may be reported as GPU validation.
Elapsed times include setup, subprocess overhead, copies, and checks; they are
**not kernel latency measurements**. Counter permissions and profiler capture
remain separate preflight results. No timing or fusion work is unlocked until
the native hardware gates pass and their evidence is reviewed.

## Cancellation, export, and limits

Each subprocess starts in its own process group. Timeout, Ctrl-C, and CLI
SIGTERM kill that group and reap the direct child before returning partial
output. This also applies to the existing preflight runner. Normal compiler
and profiler descendants inherit the group. SIGKILL, machine failure, a process
that deliberately detaches, and uninterruptible kernel/driver work are outside
this cleanup guarantee. The deadline is a workload bound with cleanup overhead,
not a hard real-time guarantee. Keep the provider stop procedure available.

After a session, copy the report and tool logs locally, verify checksums, and
retain the build configuration/compiler output. Record any failure before
retrying in a new output directory. Export the binary if needed for diagnosis,
within storage limits. Stop/delete the intended compute through the provider,
verify its state, and reconcile retained storage and actual charges separately.
Process exit and SSH disconnect do not stop billing.

The hardware gate covers this numerical matrix and the smoke subset under the
three tools. It does not prove all inputs, cross-stream lifetime safety,
multi-device behavior, paged attention, end-to-end serving, or a speedup.
Hash checks detect ordinary source/binary changes, not adversarial tampering
or a complete hermetic environment. Record GPU software/image provenance too.

## Executed status

Chunk 09 adds twelve CPU runner tests, one preflight interruption regression,
and a CTest group. Local Debug passed **82 checks**; all twelve runner tests and
seven preflight tests passed, including real timeout/SIGINT/SIGTERM child-process
cleanup. Success/failure GPU records in these tests are explicitly synthetic.
The Linux workflow now exercises a real fresh sm_86 build through host-only
mode, retains sm_89 compilation, and prints the complete host report in its log.
It passed in [run 40469386543](https://github.com/BIN0806/GPU-Inference/actions/runs/40469386543) at source `a72a702`. All seven Linux jobs passed (82 normal / 83 sanitized CTest checks; 10 reference tests and 218 CPU cases).
[Retained summary](validation/0009-linux-ci.json). No GPU host is available yet;
native hardware execution, Compute Sanitizer, and performance remain untested.

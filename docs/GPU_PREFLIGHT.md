# GPU preflight without confusing inventory and execution

The script operates on an **already provisioned host**. It does not create a
pod, purchase credit, install drivers, elevate privileges, change counter
permissions, or stop provider billing. Before a rental, record a fresh all-in
quote and cleanup procedure in [the budget ledger](BUDGET.md). The initial
checkpoint remains $20 and the total ceiling $100.

## Inspect locally before using a GPU

```sh
python3 scripts/gpu_preflight.py --output artifacts/preflight-local
```

Every output directory must be new. The JSON records selected tool versions,
GPU inventory, framework availability, repository revision/status, probe source
hashes, command outputs/return codes, and timestamps. It does not dump the entire
environment or other users' GPU processes. Inventory mode returns success for
collecting information even if tools are absent. **It always leaves
`execution_checks_passed=false` and counters `not_tested`.** Never use its exit
code alone as evidence that GPU execution works.

The actual Mac inventory in chunk 07 found no `nvidia-smi`, `nvcc`, Nsight, or
Compute Sanitizer executables. The CPU PyTorch wheel reported no CUDA runtime
and `cuda_available=false`. No device probe was executed. Preflight fixture
tests cover failure/timeout/permission classification; their strings are not
hardware measurements.

## Run on a selected development image

Use a CUDA **development** image with nvcc/headers, a supported host compiler,
matching PyTorch and Triton, and the desired debug tools. Record the image digest
and resolved packages there; the local CPU locks are not GPU environment locks.
Do not source-build PyTorch during paid time. The toolkit tuple is not selected
or verified by this chunk.

```sh
# Example target only: verify the rented device's compute capability first.
python3 scripts/gpu_preflight.py --output artifacts/preflight-gpu \
  --run-gpu --cuda-arch 86 --max-seconds 600
```

The explicit architecture is passed as one nvcc argument. Use the actual target,
not a value copied blindly from this example. The script then:

1. Compiles [gpu_smoke.cu](../tools/gpu_smoke.cu) into the output directory.
   The tiny 257-element output kernel checks its partial final block, launch
   status, execution completion, copied values, and explicit device cleanup.
2. Runs native CUDA, a PyTorch CUDA expression, a direct Triton kernel, and a
   `torch.compile` CUDA function. Each needs its own successful marker after
   synchronization and exact value checks. Package presence is insufficient.
3. Runs Compute Sanitizer memcheck and requires both correct output and a zero
   error summary. A nonzero error-exit code fails the execution gate.
4. Launches the same native kernel under Nsight Compute. A successful version
   command does not establish counter permissions. The script records verified,
   permission-denied, missing-tool, or failed status from the actual probe.
5. Attempts a CUDA timeline with Nsight Systems and requires an actual report
   file plus the successful child result to record capture as verified.

All five execution prerequisites (native, PyTorch, Triton, compile, memcheck)
must pass for `execution_checks_passed=true` and exit code zero in execution
mode. Counter/timeline status remains separate. Missing counters can still leave
a usable correctness environment; record that limitation and use unprofiled
CUDA-event timings and available timelines later. Do not label modeled traffic
as measured DRAM bandwidth.

Per-command timeouts are bounded by the remaining overall budget (600 seconds
by default, configurable up to 1800). A timeout kills the command's process
group, including compiler/profiler descendants, then retains the diagnostic
record. It does **not** delete a rental or volume. Export artifacts, verify their
local copies, stop/delete resources through the provider, and verify billing
state separately. Do not infer cleanup from an SSH disconnect or process exit.

Ctrl-C and CLI SIGTERM now use the same process-group cleanup and retain partial
command output, returning 130 with status `interrupted`. They stop later probes.
SIGKILL, machine failure, detached processes, and uninterruptible driver work
remain outside the guarantee; see [native session limits](CUDA_VALIDATION.md).

## Interpret compatibility evidence carefully

The driver capability reported by `nvidia-smi`, nvcc's installed compiler
version, and `torch.version.cuda` are different facts. A PyTorch wheel can work
while native extension compilation fails. NVIDIA's family-level compatibility
rules have PTX/JIT and feature caveats; an actual compile/launch is the useful
check. [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Counter permission behavior depends on the host driver/container setup and has
evolved across driver versions. `ERR_NVGPUCTRPERM` is a permissions finding;
container root by itself is not a portable guarantee. This script never applies
host permission changes. Use the provider's supported configuration or continue
with the documented profiling fallback.
[NVIDIA counter-permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).

## Current evidence boundary

The inventory path and orchestration failure cases have executed on CPU. Native
smoke compilation, CUDA launches, Triton, `torch.compile` on CUDA, memcheck,
counters, and timeline capture are **not yet executed on hardware** in this
project. The CUDA smoke source is preparation, not a validated GPU result.
This script cannot establish that the implemented unfused RMSNorm control is
correct, fast, or safe across streams. Use the independent
[native validation session](CUDA_VALIDATION.md) for its smoke/full matrix and
three sanitizer gates. Native correctness does not depend on framework readiness;
the framework preflight remains relevant before their comparison work.

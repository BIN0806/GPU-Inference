# First native CUDA run on Modal

Use one L4 for the first residual-add and RMSNorm hardware check. NVIDIA lists
the L4 at compute capability 8.9, which matches one of our two CI compilation
targets. Modal advertises the L4 as a selectable GPU. Allocation in a particular
workspace is confirmed when the job starts.
[NVIDIA device table](https://developer.nvidia.com/cuda/gpus) ·
[Modal GPUs](https://modal.com/docs/guide/gpu).

## Getting to the first result

1. Install the Modal CLI if needed (`python -m pip install modal==1.6.4` in
   a local virtual environment). Complete `modal token new` in your terminal
   and browser. Verify the intended workspace with `modal profile list` and its
   connectivity with `modal app list --limit 1`. Modal requires a valid payment
   method for GPU use. Check remaining credit and recent usage in Billing.
2. From the repository root, check the entrypoint locally with
   `modal run infra/modal/native_check.py::main --help`.
3. For an authorized GPU session, use a new local output directory and run in
   the foreground. Keep the App URL that the CLI prints.

```sh
modal run infra/modal/native_check.py::main --output artifacts/modal-l4-001
```

The [adapter](../infra/modal/native_check.py) fetches the public control at
`48ed5301e1a9344f6326b6169e936c24add184a0` during a CPU image build. This pins
the source that passed compile and host CI. Update the revision intentionally
for a future experiment. The base image is the Linux amd64 manifest for
`nvidia/cuda:12.8.1-devel-ubuntu24.04`, pinned by digest. CMake and Ninja come
from Ubuntu packages. The report records the installed tool versions.
[Modal CUDA images](https://modal.com/docs/guide/cuda).

Inside the GPU function, the existing [native runner](CUDA_VALIDATION.md) builds
for architecture 89 and runs the 34-case smoke matrix, memcheck, racecheck,
synccheck, and the 218-case full matrix. The local process saves the returned
`validation.json` before checking the native exit status and source revision.
Read the first failed command and its stderr. A tool permission failure leaves
that check open. Nsight counter access requires the separate
[profiling probe](GPU_PREFLIGHT.md). Investigate timing after the correctness
checks pass.

If startup fails or the client loses the function result, inspect the App logs.
The complete report becomes durable locally only once returned. In the dashboard,
check that the ephemeral App has stopped, then reconcile the session's usage in
[the ledger](BUDGET.md). The adapter uses container scratch space and returns the
report directly to the invoking machine.

## Budget estimate on November 23, 2026

At published base rates, one L4 at $0.000222/s, one physical CPU core at
$0.0000131/s and 4 GiB RAM at $0.00000222/GiB/s total **about $0.15 for 600
seconds** ($0.1464). The function timeout is 720 seconds to leave time for
returning diagnostics. That duration at the same allocation is about $0.18.
The estimate excludes image build, startup, any extra billable usage, preemption
and taxes. Default application-error retries are explicitly disabled. Modal
can still reschedule a preempted function. Reconcile gross usage and applied
credits separately. Starter advertises $30 monthly compute credit. Available
credit and actual charges depend on the workspace.
[Modal pricing](https://modal.com/pricing) · [Preemption](https://modal.com/docs/guide/preemption).

## First execution on November 23, 2026

The pinned image built and Modal started one L4 call after about 146 seconds
in the capacity queue. The call took 13.84 seconds including native build and
checks. It exported its report and the App stopped with zero active containers.
All numerical and Compute Sanitizer gates passed. These elapsed times describe
the session and include subprocess overhead. Nsight counters and repeatable
performance timings still need measurements. [Evidence](validation/0010-modal-l4.json).

## Run the fused comparison

From a clean checkout of a published candidate commit, select its full SHA
during the CPU image build. Use a fresh output directory for each authorized run.

```sh
KVLAB_FUSED_REVISION="$(git rev-parse HEAD)" modal run infra/modal/native_check.py::main \
  --output artifacts/modal-l4-002
```

The adapter verifies the returned revision and paired timing gate. Leaving
the variable unset reproduces the original control session. Both modes use
the same one-L4 allocation and deadlines. [Timing boundaries](CUDA_VALIDATION.md#fused-comparison).

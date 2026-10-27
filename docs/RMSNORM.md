# Residual + RMSNorm: first numerical contract

Chunk 07 implements a standard-library oracle and eager PyTorch reference.
The separate [native CUDA control](CUDA_RMSNORM.md) now has its own implementation
and validation boundary. Fused CUDA, Triton normalization, and an end-to-end model
remain outside the implemented scope. GPU execution of the references remains
untested until a real device session. [ADR 0008](decisions/0008-rmsnorm-contract.md)
records the alternatives and scope.

## Supported inputs and outputs

- `x`, `residual`: contiguous dense `[M,D]`; `weight`: contiguous `[D]`.
- `M >= 0`, `1 <= D <= 8192`; all tensors share one CPU or CUDA device and dtype
  (`float16` or `float32`). CPU is the executed platform in this chunk.
- Inference only: no gradients, no overlapping nonempty input byte ranges.
  Disjoint contiguous slices of the same storage are valid. Empty batches
  return correctly shaped empty outputs; their empty address ranges do not alias.
- Quantized inputs must be finite, `abs(x), abs(residual) <= 1024`, and
  `abs(weight) <= 16`. Out-of-domain inputs raise; nothing is silently clamped.
- Epsilon is fixed to `1e-6`, represented in FP32. Mixed weight precision, BF16,
  arbitrary strides, in-place outputs, and configurable epsilon are later contracts.
- Return **both** fresh outputs `(z, y)`, same shape/dtype/device as `x`.
  Inputs are unchanged. `z` is the rounded residual used by normalization.

These bounds keep the FP32 sum of squares finite even at maximum width and
keep normalized FP16 outputs representable: mathematically
`abs(y) <= sqrt(D) * max(abs(weight))`, before small rounding effects.
They are a deliberately conservative development domain, not a claim about all
model activations or all finite values of the storage type.

```text
z_j = round_T(FP32(x_j) + FP32(residual_j))
v   = mean_j(FP32(z_j) * FP32(z_j))
y_j = round_T((FP32(z_j) * rsqrt(v + FP32(epsilon))) * FP32(weight_j))
```

The eager implementation uses FP32 tensor arithmetic for addition, square,
mean, inverse square root, and multiplication; reduction order remains a backend
implementation choice. The independent oracle observes the same input/residual
rounding, then uses Python FP64 `math.fsum`, square root, and scaling before the
final storage cast. It does not call PyTorch to compute expected outputs.

## Why returning the right residual is insufficient

A fused implementation can return rounded `z` yet accidentally normalize a live
unrounded register value. Broad tolerances may not expose this semantic change.
The targeted FP16 case is therefore tested independently:

```text
x      = [1, 2, 3]
r      = [2^-11, 2^-10, 2^-9]
weight = [1, 1, 1]
z      = [1, 2, 3.001953125]
y      = [0.462646484375, 0.92529296875, 1.388671875]
```

An unrounded-residual calculation gives first values `0.462890625` and
`0.92578125`. Those differences fit the ordinary broad FP16 tolerance, so
contract-specific examples are necessary alongside random numerical checks.
The exact-output assertion here tests this CPU reference case; it is not a
promise of bitwise normalized-output agreement for every GPU reduction order.

RMSNorm does not subtract a mean, unlike LayerNorm. A one-element nonzero row
therefore remains approximately its sign times its weight. Epsilon prevents a
zero row from dividing by zero. PyTorch's built-in RMSNorm has its own epsilon
and dtype behavior; it is not silently substituted for this two-output residual
contract. [PyTorch RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html),
[numerical accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

## Files and execution

- [oracle.py](../python/kvlab/oracle.py): explicit IEEE FP16/FP32 quantization,
  rounded residual, FP64 oracle and error metrics; standard library only.
- [reference.py](../python/kvlab/reference.py): checked wrapper and separate
  prevalidated tensor math. Value checks synchronize a CUDA device if called there.
- [correctness.py](../python/kvlab/correctness.py): seeded cases, residual bit
  checks, per-element output thresholds, and diagnostic metrics.
- [matrix runner](../scripts/check_rmsnorm_reference.py): immutable output
  directory, source/version metadata, active-case failure context, and JSON results.

```sh
# Dependency-free oracle and preflight contracts also run in CTest.
PYTHONPATH=python python3 -m unittest discover -s tests/python -p test_rmsnorm_oracle.py

# macOS ARM64, Python 3.12; use the Linux CPU lock on Linux x86-64 instead.
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r scripts/requirements-rmsnorm-macos.txt
PYTHONPATH=python .venv/bin/python -m unittest discover -s tests/python -p 'test_rmsnorm_*.py' -v
.venv/bin/python scripts/check_rmsnorm_reference.py --output artifacts/rmsnorm-cpu
```

CPU-only test locks must not replace a remote CUDA PyTorch environment. The
GPU toolkit/framework/Triton tuple will be selected and verified separately.
See [dependencies](DEPENDENCIES.md). The checked wrapper is a correctness
boundary; do not time its finite/range scans as a GPU operator. A future
benchmark validates outside the timed region and calls `residual_rmsnorm_math`
under inference mode. An unchecked function is not permission to ignore its
preconditions. CPU `torch.compile` performance is not measured in this chunk.

## Correctness matrix and observed results

Ordinary seeded inputs cover `M = 0,1,2,31,32,33,128` and
`D = 1,3,31,32,33,127,128,129,768,1024,4096,8192`, both dtypes. Additional tests
use zero, exact cancellation, tiny values, bounded large values, and rounding
patterns at five widths. Total: **218 cases** (109 per dtype), separate from
boundary/alias/validation unit tests.

Residuals must match storage bits. Normalized outputs use per-element
`abs(actual-reference) <= atol + rtol*abs(reference)` with FP16 `(2e-3,2e-3)`
and FP32 `(1e-5,1e-4)`. Nonfinite comparisons fail. Report maximum absolute
error and `sqrt(sum(error²)/max(sum(reference²),1e-30))`; the latter is a
normalized RMS error with an explicit zero-reference floor. An empty comparison
has zero errors and no worst coordinate. Thresholds were not widened to pass.

Local CPU run: PyTorch 2.14.1, Python 3.12, Apple M3 Pro/macOS ARM64.
All 218 cases passed; all residuals matched bits. Largest absolute output error:
FP16 **0.001953125**, FP32 **9.5367431640625e-7**. Largest per-case normalized
RMS error: FP16 **1.2127999441234786e-4**, FP32 **8.273038858220704e-8**.
These are numerical errors, not runtime or GPU performance results. Complete
case records are in [the local CPU result](validation/0007-rmsnorm-cpu.json).
Linux x86-64 / Python 3.12.3 / PyTorch 2.14.1+cpu also passes all 218 cases in
[run 40270856228](https://github.com/BIN0806/GPU-Inference/actions/runs/40270856228). Absolute error maxima and the FP16 normalized RMS maximum match the local
run; Linux's largest per-case FP32 normalized RMS error is
`9.447669758998605e-8`. The small backend difference remains within the same
unchanged tolerance. [Linux summary](validation/0007-linux-ci.json) retains the
executed job links and metrics; the Linux per-case JSON was not uploaded.

The optional NumPy bridge is absent from these small test environments, so
PyTorch emits a startup warning. No code uses NumPy conversion, and the warning
was retained. No GPU kernel, Compute Sanitizer, device stream behavior, or
hardware profiler result is established by this CPU matrix.

## Next implementation gate

Validate the separate [CUDA add and RMSNorm controls](CUDA_RMSNORM.md) on hardware before the fused kernel.
Make the rounded residual store/reload visible in the first version. Test empty
batches and masked odd-width tails, inspect `z` before `y`, and preserve FP32
accumulation. Then compare identical two-output work with eager, compiled,
and Triton implementations. No speedup follows merely from calling a path fused.

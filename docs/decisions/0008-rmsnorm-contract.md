# ADR 0008 — Rounded residual, two outputs, independent numerical oracle

Date: 2026-10-25. Status: implemented; local and Linux CPU validation passed ([run 40270856228](https://github.com/BIN0806/GPU-Inference/actions/runs/40270856228)). GPU paths untested.

## Decision

The first normalization operation returns `(z, y)`, with `z` rounded to the input
storage dtype **before** normalization. Use FP32 tensor arithmetic in the eager
reference, FP64 reduction/scaling in a separate standard-library oracle, and
explicit input/residual/output casts. CPU references establish expected values
before rental time is spent on indexing and reductions.

Limit the first contract to contiguous FP16/FP32 inputs, shared dtype/device,
finite bounded values, D=1..8192, M>=0, no overlapping input ranges, no gradients,
and fixed epsilon 1e-6. Reject unsupported cases instead of silently making
copies, clamping values, or returning only one output. CPU-only locks and tests
remain separate from the future GPU framework/toolkit tuple.

## Alternatives and consequences

- An unrounded FP32 residual may avoid intermediate storage/rounding, but it is
  a different operation. It cannot be used as the baseline for this contract.
- Returning only normalized output can save a write, but changes the useful
  residual state and logical traffic. All future comparisons must return both.
- Reusing tensor expressions for the oracle is convenient but can share a
  precision mistake with the implementation. A loop/`math.fsum` oracle provides
  an independent numerical path, with hand-computed and rounding-specific cases.
- Accepting every finite input risks overflowing square reductions or outputs.
  The conservative domain keeps first-stage reasoning bounded; it is not a
  general production activation guarantee.
- Validating tensor values on a GPU synchronizes. Keep the checked wrapper for
  correctness and separate prevalidated math for future operator measurements.

A discovered test-design issue reinforced the choice: broad FP16 tolerances
accepted a targeted unrounded calculation. The exact CPU witness now checks the
required rounded semantics separately. Do not loosen tolerance to erase that
contract boundary. Normalized outputs across arbitrary backend reductions still
use the declared numerical thresholds; the witness is not universal bitwise GPU
reproducibility.

## Infrastructure boundary

The preflight records tool inventory separately from actual launches and
counter access. It neither provisions a host nor changes its permissions.
Probe failures and timeouts fail closed, with process-group termination and an
overall deadline. Without an NVIDIA device, fixture tests validate orchestration
only. Native CUDA smoke source and GPU execution paths remain unverified until
run on the selected machine.

Revisit for mixed weight precision, BF16, strided or in-place operation,
configurable epsilon, a wider value domain, and the optimized reduction order.
Each requires an explicit contract/test update. See [RMSNorm specification](../RMSNORM.md),
[preflight guide](../GPU_PREFLIGHT.md), and engineering entry 0011.

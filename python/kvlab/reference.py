"""Eager PyTorch reference for the explicitly rounded, two-output contract."""
import torch

from .oracle import EPSILON, INPUT_BOUND, MAX_WIDTH, WEIGHT_BOUND


def validate_inputs(x, residual, weight):
    """Debug/correctness boundary; tensor-value checks synchronize CUDA if used there."""
    tensors = (x, residual, weight)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("x, residual and weight must be tensors")
    if x.layout != torch.strided or any(t.layout != torch.strided for t in tensors):
        raise ValueError("only dense strided tensors are supported")
    if x.device.type not in ("cpu", "cuda") or any(t.device != x.device for t in tensors):
        raise ValueError("inputs must share one CPU or CUDA device")
    if x.dtype not in (torch.float16, torch.float32) or any(t.dtype != x.dtype for t in tensors):
        raise ValueError("inputs must share float16 or float32 dtype")
    if x.ndim != 2 or residual.shape != x.shape or weight.ndim != 1:
        raise ValueError("expected x/residual [M,D] and weight [D]")
    if not 1 <= x.shape[1] <= MAX_WIDTH or weight.numel() != x.shape[1]:
        raise ValueError("unsupported width or weight shape")
    if any(not t.is_contiguous() or t.requires_grad for t in tensors):
        raise ValueError("inputs must be contiguous inference tensors without gradients")
    # Only nonempty contiguous byte intervals overlap; empty tensors can share address zero.
    intervals = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                 for t in tensors if t.numel()]
    for i, (start, end) in enumerate(intervals):
        if any(start < other_end and other_start < end for other_start, other_end in intervals[i + 1:]):
            raise ValueError("input storage ranges must not overlap")
    for tensor, bound in zip(tensors, (INPUT_BOUND, INPUT_BOUND, WEIGHT_BOUND)):
        if not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > bound).any()):
            raise ValueError("nonfinite input or value outside the documented bound")


def residual_rmsnorm_math(x, residual, weight):
    """Unchecked math for prevalidated inputs; suitable for a later compile comparison.

    Do not include validate_inputs' synchronizing value scans in operator timings.
    Both outputs are required; z must be materialized with storage rounding.
    """
    z = (x.float() + residual.float()).to(x.dtype)
    rounded = z.float()
    inverse_rms = torch.rsqrt((rounded * rounded).mean(dim=-1, keepdim=True) + EPSILON)
    y = ((rounded * inverse_rms) * weight.float()).to(x.dtype)
    return z, y


@torch.inference_mode()
def residual_rmsnorm(x, residual, weight):
    validate_inputs(x, residual, weight)
    return residual_rmsnorm_math(x, residual, weight)

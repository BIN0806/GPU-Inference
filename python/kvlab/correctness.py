"""Shared correctness cases and diagnostics, never an operator timing harness."""
import torch

from .oracle import error_metrics, residual_rmsnorm_oracle
from .reference import residual_rmsnorm

ROWS = (0, 1, 2, 31, 32, 33, 128)
WIDTHS = (1, 3, 31, 32, 33, 127, 128, 129, 768, 1024, 4096, 8192)
TOLERANCES = {"float16": (2e-3, 2e-3), "float32": (1e-5, 1e-4)}


def make_case(rows, width, dtype, pattern="random", seed=701, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((rows, width), generator=generator) * 6 - 3
    residual = torch.rand((rows, width), generator=generator) * 6 - 3
    weight = torch.rand(width, generator=generator) * 4 - 2
    if pattern == "zero":
        x.zero_(); residual.zero_()
    elif pattern == "cancellation":
        residual = -x
    elif pattern == "tiny":
        x *= 1e-8; residual *= 1e-8
    elif pattern == "bounded_large":
        x.fill_(1024); residual.fill_(1024)
        weight = torch.linspace(-16, 16, width)
    elif pattern == "rounding":
        x.fill_(1); residual.fill_(2 ** (-11 if dtype == torch.float16 else -24))
    elif pattern != "random":
        raise ValueError("unknown test pattern")
    # Separate allocations remain separate after conversions, including matching dtypes.
    return tuple(t.to(device=device, dtype=dtype) for t in (x, residual, weight))


def check_case(x, residual, weight):
    z, y = residual_rmsnorm(x, residual, weight)
    dtype = str(x.dtype).removeprefix("torch.")
    reference_z, reference_y = residual_rmsnorm_oracle(
        x.cpu().tolist(), residual.cpu().tolist(), weight.cpu().tolist(), dtype)
    oracle_z = torch.tensor(reference_z, dtype=x.dtype).reshape(x.shape)
    oracle_y = torch.tensor(reference_y, dtype=x.dtype).reshape(x.shape)
    bits = torch.int16 if x.dtype == torch.float16 else torch.int32
    z_exact = torch.equal(z.cpu().view(bits), oracle_z.view(bits))
    actual = y.cpu().flatten().tolist()
    expected = oracle_y.flatten().tolist()
    metrics = error_metrics(actual, expected)
    atol, rtol = TOLERANCES[dtype]
    failures = sum(abs(a - b) > atol + rtol * abs(b) for a, b in zip(actual, expected))
    record = {"dtype": dtype, "rows": x.shape[0], "width": x.shape[1],
              "residual_bitwise_equal": z_exact, "failed_y_elements": failures,
              "atol": atol, "rtol": rtol, **metrics}
    if not z_exact or failures:
        raise AssertionError(record)
    return record

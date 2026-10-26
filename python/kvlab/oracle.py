"""Independent, standard-library residual/RMSNorm oracle; no tensor operations."""
import math
import struct

EPSILON = 1e-6
MAX_WIDTH = 8192
INPUT_BOUND = 1024.0
WEIGHT_BOUND = 16.0


def quantize(value, dtype):
    """Round to an IEEE storage format and return its exactly representable Python float."""
    formats = {"float16": "<e", "float32": "<f"}
    if dtype not in formats:
        raise ValueError("supported storage types: float16, float32")
    try:
        return struct.unpack(formats[dtype], struct.pack(formats[dtype], float(value)))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("value is not representable in the storage type") from error


def _input(value, dtype, bound):
    value = quantize(value, dtype)
    if not math.isfinite(value) or abs(value) > bound:
        raise ValueError("input is nonfinite or outside the documented bound")
    return value


def residual_rmsnorm_oracle(x, residual, weight, dtype):
    """Return fresh nested (z, y) lists, rounding residual before an FP64 reduction.

    Width comes from weight, so an empty batch retains its shape contract.
    Bounds apply after inputs are quantized to their declared storage dtype.
    This is a value oracle, not a CPU performance baseline or an aliasing checker.
    """
    width = len(weight)
    if not 1 <= width <= MAX_WIDTH or len(x) != len(residual):
        raise ValueError("invalid batch or width")
    weights = [_input(v, dtype, WEIGHT_BOUND) for v in weight]
    if any(len(row) != width for row in x) or any(len(row) != width for row in residual):
        raise ValueError("every input row must match the weight width")
    z, y = [], []
    epsilon = quantize(EPSILON, "float32")
    for left, right in zip(x, residual):
        # Even the oracle observes FP32 addition and storage rounding of z.
        row = [quantize(quantize(_input(a, dtype, INPUT_BOUND) +
                                _input(b, dtype, INPUT_BOUND), "float32"), dtype)
               for a, b in zip(left, right)]
        denominator = math.sqrt(math.fsum(v * v for v in row) / width + epsilon)
        z.append(row)
        y.append([quantize((v / denominator) * w, dtype) for v, w in zip(row, weights)])
    return z, y


def error_metrics(actual, expected):
    """Flattened-output error metrics; nonfinite values fail rather than hiding in NaN."""
    if len(actual) != len(expected):
        raise ValueError("output lengths differ")
    if any(not math.isfinite(v) for v in list(actual) + list(expected)):
        raise ValueError("nonfinite comparison")
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    denominator = math.fsum(v * v for v in expected)
    squared_error = math.fsum(e * e for e in errors)
    return {"max_abs_error": max(errors, default=0.0),
            "normalized_rms_error": math.sqrt(squared_error / max(denominator, 1e-30)),
            "worst_flat_index": max(range(len(errors)), key=errors.__getitem__) if errors else None}

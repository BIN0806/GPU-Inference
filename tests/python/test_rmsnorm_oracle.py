import math
import unittest

from kvlab.oracle import EPSILON, error_metrics, quantize, residual_rmsnorm_oracle


class OracleTest(unittest.TestCase):
    def test_quantization_ties_and_float32_addition(self):
        self.assertEqual(quantize(1 + 2 ** -11, "float16"), 1)
        self.assertEqual(quantize(1 + 3 * 2 ** -11, "float16"), 1 + 2 ** -9)
        self.assertEqual(quantize(1 + 2 ** -24, "float32"), 1)
        z, _ = residual_rmsnorm_oracle([[1]], [[2 ** -11]], [1], "float16")
        self.assertEqual(z, [[1]])
        z, y = residual_rmsnorm_oracle([[1, 2, 3]], [[2 ** -11, 2 ** -10, 2 ** -9]],
                                      [1, 1, 1], "float16")
        self.assertEqual(z, [[1, 2, 3.001953125]])
        self.assertEqual(y, [[0.462646484375, 0.92529296875, 1.388671875]])

    def test_hand_computed_rows(self):
        z, y = residual_rmsnorm_oracle([[3, 4]], [[0, 0]], [1, 1], "float32")
        self.assertEqual(z, [[3, 4]])
        denominator = math.sqrt(12.5 + quantize(EPSILON, "float32"))
        self.assertEqual(y, [[quantize(3 / denominator, "float32"), quantize(4 / denominator, "float32")]])
        _, y = residual_rmsnorm_oracle([[-3]], [[-1]], [2], "float32")
        self.assertAlmostEqual(y[0][0], -2, places=6)

    def test_zero_cancellation_and_empty_batch(self):
        for dtype in ("float16", "float32"):
            self.assertEqual(residual_rmsnorm_oracle([], [], [1, 1, 1], dtype), ([], []))
            z, y = residual_rmsnorm_oracle([[1, -2, 3]], [[-1, 2, -3]], [1, 2, -3], dtype)
            self.assertEqual(z, [[0, 0, 0]])
            self.assertEqual(y, [[0, 0, 0]])

    def test_reject_invalid_shape_dtype_and_values(self):
        cases = [([], [], [], "float32"), ([[1]], [], [1], "float32"),
                 ([[1, 2]], [[0]], [1], "float32"), ([[1]], [[0]], [1], "bfloat16"),
                 ([[math.nan]], [[0]], [1], "float32"), ([[math.inf]], [[0]], [1], "float16"),
                 ([[2048]], [[0]], [1], "float32"), ([[1]], [[0]], [32], "float32")]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                residual_rmsnorm_oracle(*args)

    def test_metrics_do_not_accept_nonfinite_or_wrong_length(self):
        self.assertEqual(error_metrics([], []), dict(max_abs_error=0, normalized_rms_error=0, worst_flat_index=None))
        self.assertEqual(error_metrics([0, 3], [0, 2])["max_abs_error"], 1)
        for a, b in [([math.nan], [0]), ([0], [math.inf]), ([], [0])]:
            with self.assertRaises(ValueError):
                error_metrics(a, b)


if __name__ == "__main__":
    unittest.main()

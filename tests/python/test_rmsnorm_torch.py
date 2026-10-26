import unittest

import torch

from kvlab.correctness import check_case, make_case
from kvlab.reference import residual_rmsnorm


class TorchReferenceTest(unittest.TestCase):
    def test_normalization_observes_the_rounded_residual(self):
        x = torch.tensor([[1, 2, 3]], dtype=torch.float16)
        r = torch.tensor([[2 ** -11, 2 ** -10, 2 ** -9]], dtype=torch.float16)
        z, y = residual_rmsnorm(x, r, torch.ones(3, dtype=torch.float16))
        self.assertEqual(z.tolist(), [[1, 2, 3.001953125]])
        # An unrounded residual produces [0.462890625, 0.92578125, 1.388671875].
        # The broad ordinary-input tolerance alone would not expose that semantic change.
        self.assertEqual(y.tolist(), [[0.462646484375, 0.92529296875, 1.388671875]])

    def test_shapes_dtypes_and_adversarial_inputs(self):
        for dtype in (torch.float16, torch.float32):
            for rows, width in ((0, 3), (1, 1), (2, 33), (3, 129), (1, 8192)):
                for pattern in ("random", "zero", "cancellation", "tiny", "bounded_large", "rounding"):
                    with self.subTest(dtype=dtype, rows=rows, width=width, pattern=pattern):
                        check_case(*make_case(rows, width, dtype, pattern))

    def test_outputs_are_fresh_inputs_unchanged_and_gradients_disabled(self):
        inputs = make_case(2, 3, torch.float32)
        before = [t.clone() for t in inputs]
        z, y = residual_rmsnorm(*inputs)
        self.assertEqual(len({t.data_ptr() for t in (*inputs, z, y)}), 5)
        self.assertFalse(z.requires_grad or y.requires_grad)
        for original, saved in zip(inputs, before):
            self.assertTrue(torch.equal(original, saved))

    def test_reject_aliases_partial_overlaps_and_weight_overlap(self):
        x, r, w = make_case(2, 3, torch.float32)
        buffer = torch.arange(9, dtype=torch.float32)
        cases = [(x, x, w), (buffer[:6].reshape(2, 3), buffer[3:].reshape(2, 3), w),
                 (x, r, x[0])]
        for inputs in cases:
            with self.subTest(), self.assertRaises(ValueError):
                residual_rmsnorm(*inputs)
        # Disjoint contiguous slices sharing a storage allocation are valid.
        storage = torch.zeros(12)
        residual_rmsnorm(storage[:6].reshape(2, 3), storage[6:].reshape(2, 3), w)

    def test_invalid_contracts_fail_explicitly(self):
        x, r, w = make_case(2, 3, torch.float32)
        cases = [(x.t(), r.t(), torch.ones(2)), (x, r.to(torch.float16), w),
                 (x.double(), r.double(), w.double()), (x, r[:1], w), (x, r, w[:2]),
                 (x.clone().requires_grad_(), r, w), (x, r, w.reshape(1, 3)),
                 (torch.empty(1, 0), torch.empty(1, 0), torch.empty(0)),
                 (torch.empty(1, 8193), torch.empty(1, 8193), torch.empty(8193)),
                 (torch.ones(2, 3, device="meta"), r, w)]
        for inputs in cases:
            with self.subTest(), self.assertRaises(ValueError):
                residual_rmsnorm(*inputs)
        with self.assertRaises(TypeError):
            residual_rmsnorm([[1]], r, w)
        for value in (float("nan"), float("inf"), 2048):
            bad = x.clone(); bad[0, 0] = value
            with self.assertRaises(ValueError):
                residual_rmsnorm(bad, r, w)
        with self.assertRaises(ValueError):
            residual_rmsnorm(x, r, torch.full_like(w, 32))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

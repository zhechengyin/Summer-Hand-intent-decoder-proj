"""Small tests for fixed selection, parity gates and window independence."""

import unittest

import numpy as np
import torch

from indy_loco.models.mingru_b.model import ModelB

from .export import FOLD, SESSION, SequentialWindow, error_summary, evenly_spaced_bins


class ExportTests(unittest.TestCase):
    def test_fixed_checkpoint_identity(self):
        self.assertEqual((SESSION, FOLD), ("indy_20160622_01", 1))

    def test_prespecified_sampling_is_unique_and_covers_endpoints(self):
        bins = np.arange(100, 5100, dtype=np.int64)
        selected = evenly_spaced_bins(bins, 128)
        self.assertEqual(len(np.unique(selected)), 128)
        np.testing.assert_array_equal(selected[[0, -1]], bins[[0, -1]])
        with self.assertRaises(ValueError):
            evenly_spaced_bins(bins, 5100)

    def test_parity_gate_rejects_large_or_nonfinite_errors(self):
        reference = np.ones((4, 1, 2), dtype=np.float32)
        self.assertTrue(error_summary(reference, reference)["passed"])
        self.assertFalse(error_summary(reference, reference + 0.01)["passed"])
        with self.assertRaises(ValueError):
            error_summary(reference, np.full_like(reference, np.nan))

    def test_wrapper_preserves_unique_parameters_and_resets(self):
        torch.manual_seed(43)
        torch.set_num_threads(2)
        model = ModelB().eval()
        wrapper = SequentialWindow(model).eval()
        self.assertEqual(sum(p.numel() for p in wrapper.parameters()), 374402)
        a, b = torch.randn(1, 192, 50), torch.randn(1, 192, 50)
        with torch.inference_mode():
            first = wrapper(a)
            wrapper(b)
            torch.testing.assert_close(first, wrapper(a), rtol=0, atol=0)
            torch.testing.assert_close(first, model(a), rtol=1e-4, atol=2e-4)
        self.assertEqual(tuple(first.shape), (1, 1, 2))


if __name__ == "__main__":
    unittest.main()

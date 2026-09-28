"""Run with python -m unittest indy_loco.models.mingru_b.test_package."""

import json
import unittest

import torch

from .load import ROOT, load_fold, predict_velocity


class PackageTests(unittest.TestCase):
    def test_all_folds_load_and_match_sequential_reference(self):
        torch.set_num_threads(2)
        torch.manual_seed(43)
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(len(manifest["folds"]), 30)
        self.assertEqual(
            len({(r["session"], r["fold"]) for r in manifest["folds"]}), 30
        )
        x = torch.randn(2, 192, 50)
        for row in manifest["folds"]:
            with self.subTest(session=row["session"], fold=row["fold"]):
                model, scalers = load_fold(row["session"], row["fold"])
                self.assertEqual(sum(p.numel() for p in model.parameters()), 374402)
                with torch.inference_mode():
                    prediction = model(x)
                    torch.testing.assert_close(
                        prediction, model.forward_sequential(x), rtol=2e-4, atol=2e-5
                    )
                    torch.testing.assert_close(
                        prediction,
                        model.forward_reference(x)[:, -1:],
                        rtol=2e-4,
                        atol=2e-5,
                    )
                    model(torch.randn_like(x))
                    torch.testing.assert_close(prediction, model(x), rtol=0, atol=0)
                self.assertTrue(
                    torch.isfinite(predict_velocity(model, x, scalers)).all()
                )


if __name__ == "__main__":
    unittest.main()

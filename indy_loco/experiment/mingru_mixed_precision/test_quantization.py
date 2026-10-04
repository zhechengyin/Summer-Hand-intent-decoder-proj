"""Synthetic operator/accounting tests; no trained checkpoint or dataset used."""
import unittest
import torch
from torch import nn
from indy_loco.models.mingru_b.model import ModelB
from .quantization import (ReferenceOp, apply_plan, dequantize_weight, ledger,
                           placement_scenarios, plans, quantize_weight)


class QuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(43)

    def test_weight_quantizer(self):
        w = torch.randn(8, 7)
        w[0] = 0
        q, scale = quantize_weight(w)
        self.assertEqual(q.dtype, torch.int8)
        self.assertTrue(bool(((dequantize_weight(q, scale)-w).abs() <= scale[:, None]*.50001).all()))
        self.assertTrue(bool((q[0] == 0).all()))
        self.assertEqual(float(scale[0]), 1.)

    def test_integer_linear(self):
        source = nn.Linear(11, 7)
        op = ReferenceOp(source, "w8a8_reference", 3.)
        x = torch.randn(2, 5, 11) * 4  # includes saturation outside calibration
        qx = torch.round(x / op.input_scale).clamp(-127, 127).long()
        accum = (qx.unsqueeze(-2) * op.qweight.long()[None, None]).sum(-1)
        expected = accum.float() * (op.input_scale * op.weight_scales) + op.bias
        torch.testing.assert_close(op(x), expected, rtol=0, atol=0)

    def test_integer_depthwise(self):
        source = nn.Conv1d(4, 4, 5, groups=4)
        op = ReferenceOp(source, "w8a8_reference", 2.)
        x = torch.randn(2, 4, 12)
        qx = torch.round(x/op.input_scale).clamp(-127,127).long()
        accum = (qx.unfold(-1,5,1) * op.qweight.long()[:,0][None,:,None,:]).sum(-1)
        expected = accum.float() * (op.input_scale*op.weight_scales)[None,:,None] + op.bias[None,:,None]
        torch.testing.assert_close(op(x), expected, rtol=0, atol=0)

    def test_weight_only(self):
        source = nn.Linear(7, 3)
        op = ReferenceOp(source, "w8a32")
        x = torch.randn(2,7)
        expected = torch.nn.functional.linear(x, dequantize_weight(op.qweight, op.weight_scales), op.bias)
        torch.testing.assert_close(op(x), expected, rtol=0, atol=0)

    def test_plans_and_memory(self):
        model = ModelB().eval()
        all_plans = plans()
        self.assertEqual(len(all_plans), 31)
        self.assertEqual(len({p["name"] for p in all_plans}),31)
        fp = ledger(model, all_plans[0])
        self.assertEqual(fp["weight_and_scale_bytes"],1497608)
        small = ledger(model, all_plans[-1])
        self.assertLess(small["weight_and_scale_bytes"],420000)
        for allocation in placement_scenarios(small):
            self.assertLessEqual(allocation["onchip_weight_bytes"],allocation["net_weight_budget_kib"]*1024)
            self.assertEqual(allocation["onchip_weight_bytes"]+allocation["sdram_weight_bytes"],small["aligned_weight_and_scale_bytes"])
            self.assertGreater(allocation["sdram_weight_bytes"],0)

    def test_model_variants_and_source_unchanged(self):
        model = ModelB().eval()
        state = {k:v.clone() for k,v in model.state_dict().items()}
        maxima = {n:10. for n,m in model.named_modules() if isinstance(m,(nn.Linear,nn.Conv1d))}
        x = torch.randn(2,192,50)
        with torch.inference_mode():
            for plan in plans():
                candidate = apply_plan(model,plan,maxima)
                result = candidate(x)
                self.assertEqual(result.shape,(2,1,2))
                self.assertTrue(bool(torch.isfinite(result).all()))
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,state[name],rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()

"""Explicit reference quantization, not a native INT8 latency implementation."""
import copy
import itertools

import torch
from torch import nn
from torch.nn import functional as F

GROUPS = ("encoder", "mingru", "ffn", "head")


def group(name):
    if name.startswith(("stem", "frontend")):
        return "encoder"
    if ".mixer." in name:
        return "mingru"
    if ".ffn." in name:
        return "ffn"
    if name.startswith("head"):
        return "head"
    return "normalization"


def plans():
    result = [{"name": "fp32", "mode": "fp32", "groups": []}]
    for mode in ("w8a32", "w8a8_reference"):
        for bits in itertools.product((False, True), repeat=len(GROUPS)):
            selected = [g for g, enabled in zip(GROUPS, bits) if enabled]
            if selected:
                result.append({"name": mode + "_" + "_".join(selected),
                               "mode": mode, "groups": selected})
    return result


def quantize_weight(weight):
    if not torch.isfinite(weight).all():
        raise ValueError("Nonfinite weight")
    shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    maxima = weight.detach().reshape(weight.shape[0], -1).abs().amax(1)
    # Zero channels get scale1, preserving exact zero without division by zero.
    scales = torch.where(maxima > 0, maxima / 127., torch.ones_like(maxima))
    q = torch.round(weight.detach() / scales.reshape(shape)).clamp(-127, 127).to(torch.int8)
    return q, scales


def dequantize_weight(q, scales):
    return q.float() * scales.reshape((q.shape[0],) + (1,) * (q.ndim - 1))


class ReferenceOp(nn.Module):
    """Stored INT8 weights, FP32 bias; integer arithmetic emulated in FP32.

    W8A8 accumulates integer-valued products as FP32: this model's K<=256
    keeps |sum| <= 256*127*127 < 2**24, so integer accumulation is exact.
    Rescaling/bias/output are FP32, NOT a fully-integer MCU kernel contract.
    W8A32 dequantizes weights for the FP32 operator on each call.
    Neither mode's host timing is an INT8 deployment latency estimate.
    """
    def __init__(self, source, mode, activation_max=0.):
        super().__init__()
        q, scales = quantize_weight(source.weight)
        self.register_buffer("qweight", q)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("bias", source.bias.detach().clone() if source.bias is not None else None)
        self.register_buffer("input_scale", torch.tensor(max(float(activation_max) / 127., 1e-12)))
        self.mode = mode
        self.convolution = isinstance(source, nn.Conv1d)
        if self.convolution:
            self.stride, self.padding, self.dilation, self.groups = source.stride, source.padding, source.dilation, source.groups
        if source.weight[0].numel() * 127 * 127 >= 2**24:
            raise ValueError("Integer reference accumulation exceeds FP32 exact integer range")

    def forward(self, values):
        if self.mode == "w8a32":
            weight = dequantize_weight(self.qweight, self.weight_scales)
            if self.convolution:
                return F.conv1d(values, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return F.linear(values, weight, self.bias)
        qinput = torch.round(values / self.input_scale).clamp(-127, 127)
        if self.convolution:
            accumulated = F.conv1d(qinput, self.qweight.float(), None, self.stride, self.padding, self.dilation, self.groups)
            output = accumulated * (self.weight_scales * self.input_scale)[None, :, None]
            return output if self.bias is None else output + self.bias[None, :, None]
        accumulated = F.linear(qinput, self.qweight.float(), None)
        output = accumulated * (self.weight_scales * self.input_scale)
        return output if self.bias is None else output + self.bias


def apply_plan(model, plan, activation_maxima):
    result = copy.deepcopy(model).eval()
    if plan["mode"] == "fp32":
        return result
    for name, module in list(result.named_modules()):
        if isinstance(module, (nn.Linear, nn.Conv1d)) and group(name) in plan["groups"]:
            parent, _, leaf = name.rpartition(".")
            target = result.get_submodule(parent) if parent else result
            setattr(target, leaf, ReferenceOp(module, plan["mode"], activation_maxima[name]))
    return result


def ledger(model, plan):
    modules = dict(model.named_modules())
    rows = []
    for name, parameter in model.named_parameters():
        parent, _, leaf = name.rpartition(".")
        module = modules[parent]
        quantized = (plan["mode"] != "fp32" and leaf == "weight"
                     and isinstance(module, (nn.Linear, nn.Conv1d)) and group(parent) in plan["groups"])
        metadata = parameter.shape[0] * 4 if quantized else 0
        if quantized and plan["mode"] == "w8a8_reference":
            metadata += 4  # One input scale per selected operator.
        size = parameter.numel() * (1 if quantized else 4) + metadata
        # MAC count is independent of weight precision. This is weight access
        # opportunity across time, not a prediction of actual external reads.
        uses = 1 if name.startswith(("head", "norm", "blocks.2.ffn", "blocks.2.norm2")) else 50
        rows.append({"tensor": name, "group": group(name), "parameters": parameter.numel(),
                     "precision": "int8" if quantized else "fp32", "bytes": size,
                     "aligned_bytes": ((size + 31) // 32) * 32, "time_uses": uses})
    return {"weight_and_scale_bytes": sum(r["bytes"] for r in rows),
            "aligned_weight_and_scale_bytes": sum(r["aligned_bytes"] for r in rows),
            "int8_weight_parameters": sum(r["parameters"] for r in rows if r["precision"] == "int8"),
            "tensors": rows}


def placement_scenarios(memory):
    results = []
    # These are hypothetical NET weight budgets, not claims of available SRAM.
    # Keep head tensors external to exercise SDRAM even when everything fits.
    for kib in (128, 256, 384, 512):
        available = kib * 1024
        allocations = []
        for row in sorted(memory["tensors"], key=lambda r: (-r["time_uses"], r["aligned_bytes"], r["tensor"])):
            size = row["aligned_bytes"]
            resident = 0 if row["tensor"].startswith("head.") else min(available, size)
            available -= resident
            allocations.append({"tensor": row["tensor"], "onchip_bytes": resident,
                                "sdram_bytes": size - resident, "time_uses": row["time_uses"],
                                "requires_tiling": 0 < resident < size})
        total = memory["aligned_weight_and_scale_bytes"]
        onchip = sum(r["onchip_bytes"] for r in allocations)
        external = total - onchip
        results.append({"net_weight_budget_kib": kib, "onchip_weight_bytes": onchip,
                        "sdram_weight_bytes": external, "onchip_fraction": onchip / total,
                        "external_bytes_once_per_window": external,
                        "external_bytes_if_reread_each_timestep": sum(r["sdram_bytes"] * r["time_uses"] for r in allocations),
                        "assumptions": "Head forced into SDRAM; 32-byte alignment; partial tensors need tiles. Budgets exclude activations, DMA scratch, stacks and existing firmware. No bank placement or latency verified.",
                        "allocations": allocations})
    return results

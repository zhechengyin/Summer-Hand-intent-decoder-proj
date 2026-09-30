"""Causal FP32 decoders, with no cross-window state or external kernel dependency.

References (small project-specific configurations, not pretrained replicas):
minGRU: https://arxiv.org/html/2410.01201v3, Appendix B.
Mamba-2: https://arxiv.org/abs/2405.21060, SSD and discrete recurrence;
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from indy_loco.experiment.phase16_parameter_scaling.model import (
    BASELINE,
    PairedChannelDropout,
    ScaledTCNGRU,
)

MODEL_NAMES = ("mingru", "mamba2", "transformer", "midsize_control")
EXPECTED_PARAMETERS = dict(zip(MODEL_NAMES, (88066, 82290, 87810, 86978), strict=True))


class MinGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(64, 128)

    def forward(self, values):
        candidate, gate = self.projection(values).chunk(2, dim=-1)
        # Clamp the unused log branch too, so torch.where has finite gradients.
        log_candidate = torch.where(
            candidate >= 0,
            (candidate.clamp_min(0) + 0.5).log(),
            -F.softplus(-candidate),
        )
        log_decay = -F.softplus(gate)
        log_write = -F.softplus(-gate) + log_candidate
        prefix = log_decay.cumsum(dim=1)
        # Exactly zero initial state: no additional g(0)=0.5 initial term.
        return (prefix + torch.logcumsumexp(log_write - prefix, dim=1)).exp()


class MinGRUBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.mixer = MinGRU()
        self.norm2 = nn.LayerNorm(64)
        self.ffn = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, values):
        values = values + self.mixer(self.norm1(values))
        return values + self.ffn(self.norm2(values))


def ssd_dense(x, dt, a_log, b, c, skip):
    """Differentiable SSD for the fixed short window; equivalent to zero-state scan.

    x: B,T,H,P; dt: B,T,H; b,c: B,T,N (one shared B/C group).
    Dense T-by-T implementation avoids materializing B,T,H,P,N states and
    requires no Triton. This is O(T^2), not the optimized long-sequence kernel.
    """
    decay_log = dt * (-a_log.exp())
    prefix = decay_log.transpose(1, 2).cumsum(-1)
    relative = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
    future = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(
        1
    )
    # Mask BEFORE exp to avoid overflow in the unused upper triangle.
    decay = relative.masked_fill(future, -torch.inf).exp()
    cb = torch.matmul(c, b.transpose(-1, -2))
    weights = decay * cb[:, None] * dt.transpose(1, 2).unsqueeze(-2)
    mixed = torch.matmul(weights, x.transpose(1, 2)).transpose(1, 2)
    return mixed + x * skip[None, None, :, None]


class Mamba2Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(64, 392, bias=False)
        self.conv = nn.Conv1d(256, 256, 4, groups=256, padding=3, bias=True)
        dt = torch.exp(
            torch.rand(8) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        dt = dt.clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.a_log = nn.Parameter(torch.empty(8).uniform_(1, 16).log())
        self.skip = nn.Parameter(torch.ones(8))
        self.rms_weight = nn.Parameter(torch.ones(128))
        self.out_proj = nn.Linear(128, 64, bias=False)

    def forward(self, values):
        z, xbc, dt_raw = self.in_proj(values).split((128, 256, 8), dim=-1)
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[:, :, : values.shape[1]]).transpose(
            1, 2
        )
        x, b, c = xbc.split((128, 64, 64), dim=-1)
        dt = F.softplus(dt_raw + self.dt_bias)
        x = x.reshape(*x.shape[:2], 8, 16)
        mixed = ssd_dense(x, dt, self.a_log, b, c, self.skip).flatten(2)
        # Official norm_before_gate=False: gate first, RMS normalization second.
        gated = mixed * F.silu(z)
        normalized = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + 1e-5)
        return self.out_proj(normalized * self.rms_weight)


class Mamba2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.mixer = Mamba2Mixer()

    def forward(self, values):
        return values + self.mixer(self.norm(values))


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.out = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.ffn = nn.Sequential(nn.Linear(64, 160), nn.GELU(), nn.Linear(160, 64))

    def forward(self, values):
        batch, length, _ = values.shape
        q, k, v = (
            item.reshape(batch, length, 4, 16).transpose(1, 2)
            for item in self.qkv(self.norm1(values)).chunk(3, dim=-1)
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
        future = torch.ones(
            length, length, device=values.device, dtype=torch.bool
        ).triu(1)
        weights = scores.masked_fill(future, -torch.inf).softmax(dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, 64)
        values = values + self.out(attended)
        return values + self.ffn(self.norm2(values))


class Decoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.channel_dropout = PairedChannelDropout()
        self.stem = nn.Linear(192, 64)
        self.dropout = nn.Dropout(0.1)
        layer, depth = {
            "mingru": (MinGRUBlock, 3),
            "mamba2": (Mamba2Block, 2),
            "transformer": (AttentionBlock, 2),
        }[name]
        self.blocks = nn.Sequential(*(layer() for _ in range(depth)))
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 2)
        if name == "transformer":
            position = torch.arange(50, dtype=torch.float32)[:, None]
            frequency = torch.exp(torch.arange(0, 64, 2) * (-math.log(10000.0) / 64))
            encoding = torch.zeros(50, 64)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            self.register_buffer("position", encoding)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1] != 192 or not 1 <= values.shape[2] <= 50:
            raise ValueError("Expected batch x 192 x time with 1 <= time <= 50")
        sequence = self.stem(self.channel_dropout(values).transpose(1, 2))
        if self.name == "transformer":
            sequence = sequence + self.position[: values.shape[2]]
        sequence = self.blocks(self.dropout(sequence))
        return self.head(self.norm(sequence))


def build_model(name):
    if name == "midsize_control":
        return ScaledTCNGRU(BASELINE)
    return Decoder(name)


def parameter_groups(model, name):
    low, high = [], []
    for key, parameter in model.named_parameters():
        if name == "midsize_control":
            encoder = not key.startswith(("gru.", "head."))
        else:
            encoder = key.startswith("stem.")
        (low if encoder else high).append(parameter)
    return [
        {"params": high, "lr": 3e-4, "name": "temporal_head"},
        {"params": low, "lr": 7.5e-5, "name": "encoder_stem"},
    ]


def capacity_report(name):
    with torch.device("meta"):
        model = build_model(name)
    total = sum(p.numel() for p in model.parameters())
    if total != EXPECTED_PARAMETERS[name]:
        raise ValueError(f"Architecture count changed: {name}={total}")
    return {
        "parameters": total,
        "fp32_weight_bytes": total * 4,
        "fp32_weight_kib": total * 4 / 1024,
        "ratio_to_midsize": total / 86978,
        "parameter_groups": {
            group["name"]: {
                "parameters": sum(p.numel() for p in group["params"]),
                "lr": group["lr"],
            }
            for group in parameter_groups(model, name)
        },
        "peak_runtime_memory_bytes": None,
        "note": "Instantiated parameter count; weights exclude activations/workspace/optimizer.",
    }

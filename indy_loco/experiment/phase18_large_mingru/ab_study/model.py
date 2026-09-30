"""Small-head A/B comparison, preserving the frozen independent-window minGRU.

A changes only the output head to 128 -> 256 -> 2. B additionally inserts a
causal depthwise-k5/pointwise residual front end before the recurrent blocks.
Neither model stores state between windows. Training and sequential arithmetic
use the frozen Phase-18 blocks, including final-output computation pruning.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..model import DEPTH, INPUT_FEATURES, WIDTH, WINDOW_BINS, Recurrence
from ..model import LargeMinGRU as FrozenLargeMinGRU

MODEL_NAMES = ("mingru_a", "mingru_b")
EXPECTED_PARAMETERS = {"mingru_a": 356_866, "mingru_b": 374_402}
FRONTEND_KERNEL = 5


class CausalTemporalFrontend(nn.Module):
    """B,T,128 residual local features; only current and four prior bins are used."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.depthwise = nn.Conv1d(
            WIDTH, WIDTH, FRONTEND_KERNEL, groups=WIDTH, padding=0, bias=True
        )
        self.pointwise = nn.Linear(WIDTH, WIDTH)

    def forward(self, values):
        normalized = self.norm(values).transpose(1, 2)
        # Left padding respects the same independent-window boundary as minGRU.
        filtered = self.depthwise(F.pad(normalized, (FRONTEND_KERNEL - 1, 0)))
        return values + self.pointwise(F.gelu(filtered.transpose(1, 2)))


class LargeMinGRU(FrozenLargeMinGRU):
    """Width128, three e=2 blocks, small head; B enables the causal front end."""

    def __init__(
        self,
        name="mingru_a",
        dropout=0.1,
        channel_dropout=0.1,
        recurrence: Recurrence = "parallel",
    ):
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown A/B model {name!r}; expected {MODEL_NAMES}")
        super().__init__(dropout, channel_dropout, recurrence)
        self.name = name
        self.head = nn.Sequential(nn.Linear(WIDTH, 256), nn.GELU(), nn.Linear(256, 2))
        # Construct extra B weights AFTER every shared parameter. Same-seed A/B
        # initialization therefore matches shared stem, recurrent and head weights.
        self.frontend = (
            CausalTemporalFrontend() if name == "mingru_b" else nn.Identity()
        )

    def _forward(self, values, recurrence, last_only):
        if (
            values.ndim != 3
            or values.shape[0] < 1
            or values.shape[1] != INPUT_FEATURES
            or not 1 <= values.shape[2] <= WINDOW_BINS
            or not values.is_floating_point()
        ):
            raise ValueError(
                "Expected floating batch x 192 x time with batch >= 1 "
                "and 1 <= time <= 50"
            )
        sequence = self.stem(self.channel_dropout(values).transpose(1, 2))
        sequence = self.frontend(self.dropout(sequence))
        for index, block in enumerate(self.blocks):
            sequence = block(
                sequence,
                recurrence=recurrence,
                last_only=last_only and index == DEPTH - 1,
            )
        return self.head(self.norm(sequence))


def build_model(
    name,
    dropout=0.1,
    channel_dropout=0.1,
    recurrence: Recurrence = "parallel",
):
    return LargeMinGRU(name, dropout, channel_dropout, recurrence)


def parameter_groups(
    model,
    learning_rate=1e-3,
    encoder_lr_scale=1.0,
    weight_decay=0.01,
    head_weight_decay=None,
):
    """Legacy first two group positions, with a separate output-head decay group.

    The temporal_head group contains all recurrent blocks, final normalization,
    and B's optional front end. Only the input linear stem is encoder_stem.
    """
    for name, value in (
        ("learning_rate", learning_rate),
        ("encoder_lr_scale", encoder_lr_scale),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if head_weight_decay is None:
        head_weight_decay = weight_decay
    for name, value in (
        ("weight_decay", weight_decay),
        ("head_weight_decay", head_weight_decay),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    temporal, stem, head = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith("stem."):
            stem.append(parameter)
        elif name.startswith("head."):
            head.append(parameter)
        else:
            temporal.append(parameter)
    return [
        {
            "params": temporal,
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "name": "temporal_head",
        },
        {
            "params": stem,
            "lr": learning_rate * encoder_lr_scale,
            "weight_decay": weight_decay,
            "name": "encoder_stem",
        },
        {
            "params": head,
            "lr": learning_rate,
            "weight_decay": head_weight_decay,
            "name": "output_head",
        },
    ]


def capacity_report(name, window_bins=WINDOW_BINS):
    if type(window_bins) is not int or not 1 <= window_bins <= WINDOW_BINS:
        raise ValueError("window_bins must be an integer in [1, 50]")
    with torch.device("meta"):
        model = build_model(name)
    total = sum(parameter.numel() for parameter in model.parameters())
    if total != EXPECTED_PARAMETERS[name]:
        raise ValueError(f"Architecture count changed: {name} has {total} parameters")
    stem = INPUT_FEATURES * WIDTH
    mixer = 2 * WIDTH * WIDTH
    ffn = 4 * WIDTH * WIDTH
    head = WIDTH * 256 + 256 * 2
    frontend = FRONTEND_KERNEL * WIDTH + WIDTH * WIDTH if name == "mingru_b" else 0
    shared = window_bins * (stem + DEPTH * mixer + frontend)
    proposed = shared + DEPTH * window_bins * ffn + head
    optimized = shared + ((DEPTH - 1) * window_bins + 1) * ffn + head
    return {
        "model": name,
        "parameters": total,
        "fp32_weight_bytes": total * 4,
        "fp32_weight_kib": total * 4 / 1024,
        "fp32_weight_mb": total * 4 / 1e6,
        "fp32_weight_mib": total * 4 / 2**20,
        "ratio_to_midsize": total / 86978,
        "window_bins": window_bins,
        "width": WIDTH,
        "blocks": DEPTH,
        "ffn_expansion": 2,
        "head_widths": [WIDTH, 256, 2],
        "causal_frontend_kernel": FRONTEND_KERNEL if name == "mingru_b" else None,
        "persistent_state": False,
        "dense_macs_per_prediction": optimized,
        "dense_macs_proposed_last_head_only": proposed,
        "dense_macs_full_sequence_reference": proposed + (window_bins - 1) * head,
        "dense_macs_saved_from_proposed": proposed - optimized,
        "frontend_dense_macs_per_prediction": window_bins * frontend,
        "parameter_groups": {
            group["name"]: {
                "parameters": sum(parameter.numel() for parameter in group["params"])
            }
            for group in parameter_groups(model)
        },
        "peak_runtime_memory_bytes": None,
        "note": (
            "FP32 weights exclude activations, optimizer, workspace and alignment. "
            "Dense MACs count linear and depthwise convolution products, including "
            "padded taps; they exclude biases, normalization, nonlinearities, "
            "elementwise recurrence and memory transfers. They are not measured "
            "MCU latency. Weights are projected over all timesteps per layer; "
            "there is no persistent state or hardware SDRAM/DMA implementation."
        ),
    }

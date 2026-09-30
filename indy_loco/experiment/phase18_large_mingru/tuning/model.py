"""Training-only regularization for the frozen Phase-18 architecture.

The parent model and its checkpoints remain unchanged. Dropout after the head's
GELUs adds no parameters or evaluation arithmetic. Linear layer indices and all
state-dict keys remain compatible with the parent, including strict loading.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..model import WINDOW_BINS, Recurrence
from ..model import LargeMinGRU as FrozenLargeMinGRU
from ..model import capacity_report as frozen_capacity_report


class HeadGELUDropout(nn.GELU):
    """Parameterless GELU followed by independently sampled activation dropout."""

    def __init__(self, probability):
        super().__init__()
        self.probability = float(probability)

    def forward(self, values):
        return F.dropout(
            super().forward(values), p=self.probability, training=self.training
        )


class LargeMinGRU(FrozenLargeMinGRU):
    """Same inference graph and keys as Phase 18, with optional head dropout."""

    def __init__(
        self,
        dropout=0.1,
        channel_dropout=0.1,
        head_dropout=0.0,
        recurrence: Recurrence = "parallel",
    ):
        if not 0 <= head_dropout < 1:
            raise ValueError("head_dropout must be in [0, 1)")
        super().__init__(dropout, channel_dropout, recurrence)
        self.head_dropout = float(head_dropout)
        # Parameterless replacements consume no RNG and retain head.0/2/4 keys.
        self.head[1] = HeadGELUDropout(head_dropout)
        self.head[3] = HeadGELUDropout(head_dropout)


def build_model(
    dropout=0.1,
    channel_dropout=0.1,
    head_dropout=0.0,
    recurrence: Recurrence = "parallel",
):
    return LargeMinGRU(dropout, channel_dropout, head_dropout, recurrence)


def parameter_groups(
    model,
    learning_rate=1e-3,
    encoder_lr_scale=1.0,
    weight_decay=0.01,
    head_weight_decay=None,
):
    """Split head decay while retaining the first two legacy group positions.

    ``temporal_head`` is kept as the legacy log name for group zero; this group
    now contains only the temporal blocks and final normalization. Group two is
    ``output_head``. Head LR stays equal to temporal LR, and decay is applied to
    all parameters just as in the original recipe, including biases and norms.
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


def capacity_report(window_bins=WINDOW_BINS):
    report = frozen_capacity_report(window_bins)
    with torch.device("meta"):
        model = build_model()
    total = sum(parameter.numel() for parameter in model.parameters())
    if total != report["parameters"]:
        raise ValueError("Tuning regularization changed the frozen parameter count")
    report["parameter_groups"] = {
        group["name"]: {
            "parameters": sum(parameter.numel() for parameter in group["params"])
        }
        for group in parameter_groups(model)
    }
    report["head_dropout_training_only"] = True
    report["note"] += (
        " Optional head activation dropout is disabled in evaluation and adds "
        "no inference parameters or dense MACs."
    )
    return report

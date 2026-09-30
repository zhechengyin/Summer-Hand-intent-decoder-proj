"""Independent-window minGRU with a large last-timestep readout.

Training uses the Phase-17 positive-candidate log scan. ``forward_sequential``
uses the equivalent zero-state recurrence for deployment/reference validation.
Both paths project all timesteps in one matrix operation per block: weights are
reused across time within a layer instead of issuing a linear call per step.
Actual SDRAM placement, DMA scheduling and MCU kernels are separate work.
"""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from indy_loco.experiment.phase16_parameter_scaling.model import PairedChannelDropout

WIDTH = 128
DEPTH = 3
INPUT_FEATURES = 192
WINDOW_BINS = 50
EXPECTED_PARAMETERS = 1_211_906
Recurrence = Literal["parallel", "sequential"]


def _check_recurrence(recurrence):
    if recurrence not in ("parallel", "sequential"):
        raise ValueError("recurrence must be 'parallel' or 'sequential'")


class MinGRU(nn.Module):
    """Positive-candidate minGRU; no hidden state persists between calls."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(WIDTH, 2 * WIDTH)

    def forward(self, values, recurrence: Recurrence = "parallel", last_only=False):
        _check_recurrence(recurrence)
        # One B,T,D projection reuses each weight for the whole input sequence.
        candidate, gate = self.projection(values).chunk(2, dim=-1)
        if recurrence == "parallel":
            log_candidate = torch.where(
                candidate >= 0,
                (candidate.clamp_min(0) + 0.5).log(),
                -F.softplus(-candidate),
            )
            log_decay = -F.softplus(gate)
            log_write = -F.softplus(-gate) + log_candidate
            prefix = log_decay.cumsum(dim=1)
            # No initial term: h_0 is exactly zero, as in Phase 17.
            if last_only:
                return (
                    prefix[:, -1:]
                    + torch.logsumexp(log_write - prefix, dim=1, keepdim=True)
                ).exp()
            return (prefix + torch.logcumsumexp(log_write - prefix, dim=1)).exp()

        candidate = torch.where(candidate >= 0, candidate + 0.5, candidate.sigmoid())
        write = gate.sigmoid()
        state = torch.zeros_like(candidate[:, 0])
        states = []
        for step in range(values.shape[1]):
            state = (1.0 - write[:, step]) * state + write[:, step] * candidate[:, step]
            if not last_only:
                states.append(state)
        if last_only:
            return state.unsqueeze(1)
        return torch.stack(states, dim=1)


class MinGRUBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH)
        self.mixer = MinGRU()
        self.norm2 = nn.LayerNorm(WIDTH)
        self.ffn = nn.Sequential(
            nn.Linear(WIDTH, 2 * WIDTH), nn.GELU(), nn.Linear(2 * WIDTH, WIDTH)
        )

    def forward(self, values, recurrence: Recurrence = "parallel", last_only=False):
        mixed = self.mixer(self.norm1(values), recurrence, last_only=last_only)
        # Earlier FFN outputs in the FINAL block cannot affect the readout or
        # mixer state. Earlier blocks still need all FFN outputs for the next one.
        if last_only:
            values = values[:, -1:]
        values = values + mixed
        return values + self.ffn(self.norm2(values))


class LargeMinGRU(nn.Module):
    """Width 128, three e=2 blocks, GELU head 128 -> 768 -> 1024 -> 2.

    Input: B x 192 x T, 1 <= T <= 50. Output: B x 1 x 2, compatible with
    the frozen ``evaluate_last`` routine. Every forward starts from zero state.
    ``forward_reference`` deliberately retains discarded outputs for testing.
    """

    def __init__(
        self, dropout=0.1, channel_dropout=0.1, recurrence: Recurrence = "parallel"
    ):
        super().__init__()
        for name, probability in (
            ("dropout", dropout),
            ("channel_dropout", channel_dropout),
        ):
            if not 0 <= probability < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        _check_recurrence(recurrence)
        self.recurrence = recurrence
        self.channel_dropout = PairedChannelDropout()
        self.channel_dropout.probability = float(channel_dropout)
        self.stem = nn.Linear(INPUT_FEATURES, WIDTH)
        self.dropout = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(MinGRUBlock() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Sequential(
            nn.Linear(WIDTH, 768),
            nn.GELU(),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 2),
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
        sequence = self.dropout(sequence)
        for index, block in enumerate(self.blocks):
            sequence = block(
                sequence,
                recurrence=recurrence,
                last_only=last_only and index == DEPTH - 1,
            )
        return self.head(self.norm(sequence))

    def forward(self, values):
        return self._forward(values, self.recurrence, last_only=True)

    def forward_sequential(self, values):
        """Sequential arithmetic, still resetting each independent input window."""
        return self._forward(values, "sequential", last_only=True)

    def forward_reference(self, values, recurrence=None):
        """Unpruned B x T x 2 reference; not the production training path."""
        return self._forward(values, recurrence or self.recurrence, last_only=False)


def build_model(dropout=0.1, channel_dropout=0.1, recurrence: Recurrence = "parallel"):
    return LargeMinGRU(dropout, channel_dropout, recurrence)


def parameter_groups(model, learning_rate=1e-3, encoder_lr_scale=1.0):
    """Keep Phase-17 optimizer group ordering and names for training logs."""
    if learning_rate <= 0 or encoder_lr_scale <= 0:
        raise ValueError("learning_rate and encoder_lr_scale must be positive")
    stem, temporal = [], []
    for name, parameter in model.named_parameters():
        (stem if name.startswith("stem.") else temporal).append(parameter)
    return [
        {"params": temporal, "lr": learning_rate, "name": "temporal_head"},
        {
            "params": stem,
            "lr": learning_rate * encoder_lr_scale,
            "name": "encoder_stem",
        },
    ]


def capacity_report(window_bins=WINDOW_BINS):
    if type(window_bins) is not int or not 1 <= window_bins <= WINDOW_BINS:
        raise ValueError("window_bins must be an integer in [1, 50]")
    with torch.device("meta"):
        model = build_model()
    total = sum(parameter.numel() for parameter in model.parameters())
    if total != EXPECTED_PARAMETERS:
        raise ValueError(
            f"Architecture count changed: {total} != {EXPECTED_PARAMETERS}"
        )
    stem_per_step = INPUT_FEATURES * WIDTH
    mixer_per_step = 2 * WIDTH * WIDTH
    ffn_per_step = 4 * WIDTH * WIDTH
    head_per_prediction = WIDTH * 768 + 768 * 1024 + 1024 * 2
    temporal_macs = window_bins * (stem_per_step + DEPTH * mixer_per_step)
    proposed = temporal_macs + DEPTH * window_bins * ffn_per_step + head_per_prediction
    optimized = (
        temporal_macs
        + ((DEPTH - 1) * window_bins + 1) * ffn_per_step
        + head_per_prediction
    )
    full_reference = proposed + (window_bins - 1) * head_per_prediction
    return {
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
        "head_widths": [WIDTH, 768, 1024, 2],
        "persistent_state": False,
        "dense_macs_per_prediction": optimized,
        "dense_macs_proposed_last_head_only": proposed,
        "dense_macs_full_sequence_reference": full_reference,
        "dense_macs_saved_from_proposed": proposed - optimized,
        "parameter_groups": {
            group["name"]: {
                "parameters": sum(parameter.numel() for parameter in group["params"])
            }
            for group in parameter_groups(model)
        },
        "peak_runtime_memory_bytes": None,
        "note": (
            "FP32 weights exclude activations, optimizer, workspace and alignment. "
            "Dense MACs exclude biases, normalization, nonlinearities, recurrence "
            "elementwise operations and memory transfers; these are not MCU timings. "
            "Batched projections reuse weights over time; SDRAM DMA/tiling is separate."
        ),
    }

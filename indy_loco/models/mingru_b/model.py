"""Retained Phase18 model B; standalone inference, no experiment imports."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class PairedChannelDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.probability = 0.1

    def forward(self, values):
        if not self.training:
            return values
        keep = 1.0 - self.probability
        mask = (
            torch.empty(
                values.shape[0], 96, 1, device=values.device, dtype=values.dtype
            ).bernoulli_(keep)
            / keep
        )
        return values * torch.cat((mask, mask), dim=1)


WIDTH = 128
DEPTH = 3
INPUT_FEATURES = 192
WINDOW_BINS = 50
EXPECTED_PARAMETERS = 374_402
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


class CausalTemporalFrontend(nn.Module):
    """B,T,128 residual local features; only current and four prior bins are used."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.depthwise = nn.Conv1d(WIDTH, WIDTH, 5, groups=WIDTH, padding=0, bias=True)
        self.pointwise = nn.Linear(WIDTH, WIDTH)

    def forward(self, values):
        normalized = self.norm(values).transpose(1, 2)
        # Left padding respects the same independent-window boundary as minGRU.
        filtered = self.depthwise(F.pad(normalized, (5 - 1, 0)))
        return values + self.pointwise(F.gelu(filtered.transpose(1, 2)))


class ModelB(nn.Module):
    """Width128, three blocks, causal frontend and 128->256->2 readout.

    Input is normalized B x 192 x T (1 <= T <= 50); output B x 1 x 2.
    Each call starts with zero recurrent state.
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
        self.head = nn.Sequential(nn.Linear(WIDTH, 256), nn.GELU(), nn.Linear(256, 2))
        self.frontend = CausalTemporalFrontend()

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

    def forward(self, values):
        return self._forward(values, self.recurrence, last_only=True)

    def forward_sequential(self, values):
        """Sequential arithmetic, still resetting each independent input window."""
        return self._forward(values, "sequential", last_only=True)

    def forward_reference(self, values, recurrence=None):
        """Unpruned B x T x 2 reference; not the production training path."""
        return self._forward(values, recurrence or self.recurrence, last_only=False)

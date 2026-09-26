"""Size-configurable training model; non-architecture behavior matches Phase 13."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .session_data import CHANNEL_DROPOUT, MODEL_DROPOUT, PHYSICAL_CHANNELS


@dataclass(frozen=True)
class Architecture:
    encoder_width: int = 80
    encoder_kernel_size: int = 3
    encoder_dilations: tuple[int, ...] = (1, 2, 4, 8)
    decoder_hidden_size: int = 80
    decoder_layers: int = 1

    def __post_init__(self):
        dimensions = (
            self.encoder_width,
            self.encoder_kernel_size,
            self.decoder_hidden_size,
            self.decoder_layers,
            *self.encoder_dilations,
        )
        if not self.encoder_dilations or any(
            type(v) is not int or v <= 0 for v in dimensions
        ):
            raise ValueError(
                "Architecture dimensions and dilations must be positive integers"
            )
        if self.encoder_width < 64 or self.decoder_hidden_size < 64:
            raise ValueError("Phase 16 scales up from width/hidden size 64")
        if self.encoder_kernel_size < 3 or len(self.encoder_dilations) < 4:
            raise ValueError("Phase 16 scales up from kernel 3 / four TCN blocks")

    @property
    def encoder_layers(self):
        return len(self.encoder_dilations)

    def metadata(self):
        result = asdict(self)
        result["encoder_dilations"] = list(self.encoder_dilations)
        result["encoder_layers"] = self.encoder_layers
        result["encoder_receptive_field_bins"] = 1 + (
            self.encoder_kernel_size - 1
        ) * sum(self.encoder_dilations)
        result["input_window_bins"] = 50
        result["input_features"] = 192
        return result


BASELINE = Architecture(64, 3, (1, 2, 4, 8), 64, 1)


class PointwiseLayerNorm(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.normalization = nn.LayerNorm(features)

    def forward(self, values):
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class PairedChannelDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.probability = CHANNEL_DROPOUT

    def forward(self, values):
        if not self.training:
            return values
        keep_probability = 1.0 - self.probability
        mask = torch.empty(
            values.shape[0],
            PHYSICAL_CHANNELS,
            1,
            device=values.device,
            dtype=values.dtype,
        ).bernoulli_(keep_probability)
        mask /= keep_probability
        return values * torch.cat((mask, mask), dim=1)


class ScaledTCNGRU(nn.Module):
    def __init__(self, architecture: Architecture):
        super().__init__()
        self.architecture = architecture
        width = architecture.encoder_width
        hidden = architecture.decoder_hidden_size
        kernel = architecture.encoder_kernel_size
        self.channel_dropout = PairedChannelDropout()
        self.spatial = nn.Sequential(
            nn.Conv1d(192, width, 1), PointwiseLayerNorm(width), nn.ReLU()
        )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(width, width, kernel, padding=(kernel - 1) * d, dilation=d)
                for d in architecture.encoder_dilations
            ]
        )
        self.paddings = [(kernel - 1) * d for d in architecture.encoder_dilations]
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(MODEL_DROPOUT)
        # Preserve baseline GRU dropout=0, direction, bias and hidden reset.
        self.gru = nn.GRU(
            width,
            hidden,
            num_layers=architecture.decoder_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1] != 192:
            raise ValueError("Expected (batch, 192, time)")
        encoded = self.spatial(self.channel_dropout(values))
        for convolution, padding in zip(self.convolutions, self.paddings, strict=True):
            convolved = convolution(encoded)
            if padding:
                convolved = convolved[:, :, :-padding]
            encoded = self.activation(convolved + encoded)
        encoded, _ = self.gru(self.dropout(encoded).transpose(1, 2))
        return self.head(encoded)


def parameter_report(architecture):
    # Meta construction counts every real parameter without consuming RNG or RAM.
    with torch.device("meta"):
        model = ScaledTCNGRU(architecture)
    encoder = sum(
        p.numel()
        for n, p in model.named_parameters()
        if not n.startswith(("gru.", "head."))
    )
    decoder = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.startswith(("gru.", "head."))
    )
    total = encoder + decoder
    return {
        "encoder_parameters": encoder,
        "decoder_parameters": decoder,
        "total_parameters": total,
        "fp32_weight_bytes": total * 4,
        "fp32_weight_mib": total * 4 / 2**20,
        "ratio_to_midsize": total / 86978,
        "memory_note": "Weights only; excludes activations, workspace, alignment and firmware.",
    }

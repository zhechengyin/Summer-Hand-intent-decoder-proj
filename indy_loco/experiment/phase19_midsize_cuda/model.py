"""Same Midsize parameters/state_dict and training dropout; isolated ablations."""
import torch
import torch.nn.functional as F

from indy_loco.experiment.phase16_parameter_scaling.model import BASELINE, ScaledTCNGRU
from .ops import GRUGates, LayerNormReLU, ResidualReLU

VARIANTS = ("pytorch", "epilogue", "causal", "layernorm", "gru", "combined")


def packed_gru(gru, values):
    """All three branches retained; state resets each window, as in nn.GRU."""
    if gru.num_layers != 1 or gru.bidirectional or gru.dropout != 0 or not gru.bias:
        raise ValueError("Phase19 supports the original single-layer biased GRU only")
    b, t, _ = values.shape
    # Input projection is batched across time. cuBLAS handles all matrix products.
    projected = F.linear(values, gru.weight_ih_l0, gru.bias_ih_l0)
    # Time-major contiguous storage prevents a Bx3H copy for every time step.
    projected = projected.transpose(0, 1).contiguous()
    hidden = values.new_zeros((b, gru.hidden_size))
    states = []
    for step in range(t):
        recurrent = F.linear(hidden, gru.weight_hh_l0, gru.bias_hh_l0)
        hidden = GRUGates.apply(projected[step], recurrent, hidden)
        states.append(hidden)
    return torch.stack(states, dim=1)


class Candidate(ScaledTCNGRU):
    def __init__(self, variant):
        if variant not in VARIANTS or variant == "pytorch":
            raise ValueError(variant)
        super().__init__(BASELINE)
        self.variant = variant

    def forward(self, values):
        if values.ndim != 3 or values.shape[1] != 192:
            raise ValueError("Expected (batch,192,time)")
        v = self.variant
        encoded = self.channel_dropout(values)
        if v in ("layernorm", "combined"):
            norm = self.spatial[1].normalization
            if norm.eps != 1e-5:
                raise ValueError("Normalization epsilon changed")
            encoded = LayerNormReLU.apply(self.spatial[0](encoded), norm.weight, norm.bias)
        else:
            encoded = self.spatial(encoded)
        for convolution, padding in zip(self.convolutions, self.paddings, strict=True):
            if v in ("causal", "combined"):
                # Exact tap order, only left padding, no discarded right outputs.
                # Retain vendor convolution/backward; measure the explicit pad cost too.
                convolved = F.conv1d(F.pad(encoded, (padding, 0)),
                                    convolution.weight, convolution.bias,
                                    dilation=convolution.dilation)
                encoded = ResidualReLU.apply(convolved, encoded)
            elif v == "epilogue":
                encoded = ResidualReLU.apply(convolution(encoded), encoded)
            else:
                encoded = self.activation(convolution(encoded)[:, :, :-padding] + encoded)
        values = self.dropout(encoded).transpose(1, 2)
        if v in ("gru", "combined"):
            states = packed_gru(self.gru, values)
        else:
            states, _ = self.gru(values)
        # Keep full head identical in every arm; head pruning is outside these tests.
        return self.head(states)


def create_model(variant, initial_state):
    model = ScaledTCNGRU(BASELINE) if variant == "pytorch" else Candidate(variant)
    model.load_state_dict(initial_state, strict=True)
    return model


def optimizer_for(model):
    recurrent, encoder = [], []
    for name, parameter in model.named_parameters():
        (recurrent if name.startswith(("gru.", "head.")) else encoder).append(parameter)
    return torch.optim.AdamW([
        {"params": recurrent, "lr": 3e-4, "name": "gru_head"},
        {"params": encoder, "lr": 7.5e-5, "name": "encoder_tcn"},
    ], weight_decay=0.025)

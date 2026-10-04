"""The two user-prescribed smaller b2 architectures; independent 50-bin windows."""
import torch
from torch import nn
from indy_loco.experiment.phase18_large_mingru.model import MinGRU, MinGRUBlock
from indy_loco.experiment.phase18_large_mingru.ab_study.model import CausalTemporalFrontend
from indy_loco.experiment.phase16_parameter_scaling.model import PairedChannelDropout

ARCHITECTURES = {'w128_d2': (128, 2), 'w96_d3': (96, 3)}


class Mixer(MinGRU):
    def __init__(self, width):
        nn.Module.__init__(self)
        self.projection = nn.Linear(width, 2 * width)


class Block(MinGRUBlock):
    def __init__(self, width):
        nn.Module.__init__(self)
        self.norm1 = nn.LayerNorm(width)
        self.mixer = Mixer(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))


class Frontend(CausalTemporalFrontend):
    def __init__(self, width):
        nn.Module.__init__(self)
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, groups=width, padding=0, bias=True)
        self.pointwise = nn.Linear(width, width)


class Model(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.width, self.depth = ARCHITECTURES[architecture]
        self.architecture = architecture
        self.channel_dropout = PairedChannelDropout()
        self.channel_dropout.probability = 0.1
        self.stem = nn.Linear(192, self.width)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(Block(self.width) for _ in range(self.depth))
        self.norm = nn.LayerNorm(self.width)
        self.head = nn.Sequential(nn.Linear(self.width, 2 * self.width), nn.GELU(), nn.Linear(2 * self.width, 2))
        self.frontend = Frontend(self.width)

    def _forward(self, values, recurrence, last_only):
        if values.ndim != 3 or values.shape[1] != 192 or not 1 <= values.shape[2] <= 50:
            raise ValueError('Expected batch x 192 x time, with time <= 50')
        sequence = self.frontend(self.dropout(self.stem(self.channel_dropout(values).transpose(1, 2))))
        for index, block in enumerate(self.blocks):
            sequence = block(sequence, recurrence=recurrence, last_only=last_only and index == self.depth - 1)
        return self.head(self.norm(sequence))

    def forward(self, values):
        return self._forward(values, 'parallel', True)

    def forward_sequential(self, values):
        return self._forward(values, 'sequential', True)

    def forward_reference(self, values, recurrence='parallel'):
        return self._forward(values, recurrence, False)

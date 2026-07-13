"""Architecture builders for the lightweight-decoder experiment (ARCHITECTURE_EXPERIMENT.md).

Each builder is build(cfg, n_ch) -> nn.Module, seq2seq (per-timestep 2D velocity),
usable as harness.run(build=...). The reference TCN+GRU (build_net) is causal in
its TCN (padding=(k-1)*d then slice [:-p]) and causal in the GRU when bidir=False.
So models 1 (bidir) and 5 (causal) are just build_net with bidir True/False.
"""
from __future__ import annotations

import models.tcn_gru.best_model as M


def _act(cfg):
    import torch.nn as nn
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(cfg.get("act", "relu"), nn.ReLU)


def build_causal_tcn(cfg, n_ch):
    """Model 2: wide causal dilated Conv1D residual stack + per-timestep linear head (no GRU)."""
    import torch.nn as nn
    Act = _act(cfg)

    class CausalTCN(nn.Module):
        def __init__(self):
            super().__init__()
            F = cfg["F"]
            self.sp = nn.Sequential(nn.Conv1d(n_ch, F, 1), nn.BatchNorm1d(F), Act())
            self.convs = nn.ModuleList([nn.Conv1d(F, F, 3, padding=(3 - 1) * d, dilation=d)
                                        for d in cfg["dils"]])
            self.pads = [(3 - 1) * d for d in cfg["dils"]]
            self.bns = nn.ModuleList([nn.BatchNorm1d(F) for _ in cfg["dils"]])
            self.act = Act(); self.drop = nn.Dropout(cfg["dropout"])
            self.head = nn.Linear(F, cfg.get("n_out", 2))

        def forward(self, x):
            z = self.sp(x)
            for c, p, bn in zip(self.convs, self.pads, self.bns):
                z = self.act(bn(c(z)[:, :, :-p]) + z)          # causal residual
            return self.head(self.drop(z).transpose(1, 2))
    return CausalTCN()


def build_dws_tcn(cfg, n_ch):
    """Model 3: depthwise-separable causal TCN (MCU-efficient version of model 2)."""
    import torch.nn as nn
    Act = _act(cfg)

    class DWSTCN(nn.Module):
        def __init__(self):
            super().__init__()
            F = cfg["F"]
            self.sp = nn.Sequential(nn.Conv1d(n_ch, F, 1), nn.BatchNorm1d(F), Act())
            self.dw = nn.ModuleList([nn.Conv1d(F, F, 3, padding=(3 - 1) * d, dilation=d, groups=F)
                                     for d in cfg["dils"]])           # depthwise
            self.pw = nn.ModuleList([nn.Conv1d(F, F, 1) for _ in cfg["dils"]])  # pointwise
            self.pads = [(3 - 1) * d for d in cfg["dils"]]
            self.bns = nn.ModuleList([nn.BatchNorm1d(F) for _ in cfg["dils"]])
            self.act = Act(); self.drop = nn.Dropout(cfg["dropout"])
            self.head = nn.Linear(F, cfg.get("n_out", 2))

        def forward(self, x):
            z = self.sp(x)
            for dw, pw, p, bn in zip(self.dw, self.pw, self.pads, self.bns):
                z = self.act(bn(pw(dw(z)[:, :, :-p])) + z)
            return self.head(self.drop(z).transpose(1, 2))
    return DWSTCN()


def build_gru_only(cfg, n_ch):
    """Model 4: channel embed + GRU + head, no TCN. bidir per cfg."""
    import torch.nn as nn
    Act = _act(cfg)

    class GRUOnly(nn.Module):
        def __init__(self):
            super().__init__()
            F = cfg["F"]
            self.emb = nn.Sequential(nn.Conv1d(n_ch, F, 1), nn.BatchNorm1d(F), Act())
            self.gru = nn.GRU(F, cfg["H"], cfg["L"], batch_first=True,
                              bidirectional=cfg["bidir"],
                              dropout=cfg["dropout"] if cfg["L"] > 1 else 0.0)
            self.head = nn.Linear(cfg["H"] * (2 if cfg["bidir"] else 1), cfg.get("n_out", 2))

        def forward(self, x):
            z = self.emb(x).transpose(1, 2)
            z, _ = self.gru(z)
            return self.head(z)
    return GRUOnly()


def build_lookahead_tcngru(lookahead_steps):
    """Model 6: causal TCN+GRU that may see `lookahead_steps` future frames.

    Wraps the causal build_net (bidir=False); at forward, right-pads the input by K
    and drops the first K outputs so output[t] has consumed inputs up to t+K.
    Align targets by [:, :-K] in the caller? No -- we keep length by using the model
    on a K-padded input and returning the K-shifted stream, so shapes match and each
    prediction legitimately used K future frames (latency = K*40 ms)."""
    import torch
    import torch.nn as nn

    def build(cfg, n_ch):
        core = M.build_net({**cfg, "bidir": False}, n_ch)

        class Lookahead(nn.Module):
            def __init__(self):
                super().__init__()
                self.core = core; self.k = lookahead_steps

            def forward(self, x):                              # x (B,C,T)
                xp = nn.functional.pad(x, (0, self.k))         # pad K future frames (zeros)
                y = self.core(xp)                              # (B, T+K, n_out)
                return y[:, self.k:, :]                         # pred[t] saw inputs..t+k
        return Lookahead()
    return build


def build_tcn_lstm(cfg, n_ch):
    """TCN front-end + LSTM (vs GRU). Causal when bidir=False."""
    import torch.nn as nn
    Act = _act(cfg)

    class TCNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            F = cfg["F"]
            self.sp = nn.Sequential(nn.Conv1d(n_ch, F, 1), nn.BatchNorm1d(F), Act())
            self.convs = nn.ModuleList([nn.Conv1d(F, F, 3, padding=(3 - 1) * d, dilation=d)
                                        for d in cfg["dils"]])
            self.pads = [(3 - 1) * d for d in cfg["dils"]]
            self.act = Act(); self.drop = nn.Dropout(cfg["dropout"])
            self.lstm = nn.LSTM(F, cfg["H"], cfg["L"], batch_first=True,
                                bidirectional=cfg["bidir"],
                                dropout=cfg["dropout"] if cfg["L"] > 1 else 0.0)
            self.head = nn.Linear(cfg["H"] * (2 if cfg["bidir"] else 1), cfg.get("n_out", 2))

        def forward(self, x):
            z = self.sp(x)
            for c, p in zip(self.convs, self.pads):
                z = self.act(c(z)[:, :, :-p] + z)
            z, _ = self.lstm(self.drop(z).transpose(1, 2))
            return self.head(z)
    return TCNLSTM()


def build_lstm_only(cfg, n_ch):
    """Channel embed + LSTM + head (no CNN). Causal when bidir=False."""
    import torch.nn as nn
    Act = _act(cfg)

    class LSTMOnly(nn.Module):
        def __init__(self):
            super().__init__()
            F = cfg["F"]
            self.emb = nn.Sequential(nn.Conv1d(n_ch, F, 1), nn.BatchNorm1d(F), Act())
            self.lstm = nn.LSTM(F, cfg["H"], cfg["L"], batch_first=True,
                                bidirectional=cfg["bidir"],
                                dropout=cfg["dropout"] if cfg["L"] > 1 else 0.0)
            self.head = nn.Linear(cfg["H"] * (2 if cfg["bidir"] else 1), cfg.get("n_out", 2))

        def forward(self, x):
            z, _ = self.lstm(self.emb(x).transpose(1, 2))
            return self.head(z)
    return LSTMOnly()


def build_plain_cnn_gru(cfg, n_ch):
    """Plain (NON-dilated) causal Conv1d stack + GRU + head. Shorter receptive field
    than the dilated TCN -- tests whether the dilations matter."""
    import torch.nn as nn
    Act = _act(cfg)

    class PlainCNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            F = cfg["F"]
            self.sp = nn.Sequential(nn.Conv1d(n_ch, F, 1), nn.BatchNorm1d(F), Act())
            self.convs = nn.ModuleList([nn.Conv1d(F, F, 3, padding=2)   # kernel 3, causal
                                        for _ in cfg["dils"]])           # same depth as TCN
            self.act = Act(); self.drop = nn.Dropout(cfg["dropout"])
            self.gru = nn.GRU(F, cfg["H"], cfg["L"], batch_first=True,
                              bidirectional=cfg["bidir"],
                              dropout=cfg["dropout"] if cfg["L"] > 1 else 0.0)
            self.head = nn.Linear(cfg["H"] * (2 if cfg["bidir"] else 1), cfg.get("n_out", 2))

        def forward(self, x):
            z = self.sp(x)
            for c in self.convs:
                z = self.act(c(z)[:, :, :-2] + z)                        # causal (drop last 2)
            z, _ = self.gru(self.drop(z).transpose(1, 2))
            return self.head(z)
    return PlainCNNGRU()


def build_transformer(cfg, n_ch):
    """Causal Transformer encoder (attention) + head. Causal via subsequent mask."""
    import torch
    import torch.nn as nn

    class TF(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg["F"]
            self.emb = nn.Conv1d(n_ch, d, 1)
            self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                               dropout=cfg["dropout"], batch_first=True,
                                               activation="relu")
            self.enc = nn.TransformerEncoder(layer, max(cfg["L"], 2))
            self.head = nn.Linear(d, cfg.get("n_out", 2))

        def forward(self, x):
            z = self.emb(x).transpose(1, 2)                              # (B,T,d)
            T = z.shape[1]
            z = z + self.pos[:, :T]
            mask = nn.Transformer.generate_square_subsequent_mask(T).to(z.device)  # causal
            return self.head(self.enc(z, mask=mask))
    return TF()

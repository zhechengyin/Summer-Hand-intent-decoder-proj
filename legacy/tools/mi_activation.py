#!/usr/bin/env python
"""Which activation function is best for MI (motor imagery) decoding?

Small-sample sweep on the DECODABLE MI benchmark: eegmmidb left/right hand MI
(ds004362), a few subjects, subject-specific K-fold. Compact EEGNet-style CNN
with the activation function swapped; everything else held fixed. Reports mean
accuracy per activation (chance = 0.50).

Usage: py tools/mi_activation.py --subjects 4 --folds 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eegmmidb_probe import SFREQ, build_epochs

ACTS = ["gelu", "relu", "elu", "silu", "leaky_relu", "tanh", "mish", "selu"]


def make_act(name):
    import torch.nn as nn
    return {"gelu": nn.GELU, "relu": nn.ReLU, "elu": nn.ELU, "silu": nn.SiLU,
            "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh, "mish": nn.Mish,
            "selu": nn.SELU}[name]()


def build_cnn(n_ch, n_t, n_cls, act):
    import torch.nn as nn
    F1, D, F2 = 8, 2, 16

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, 33), padding=(0, 16), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D), make_act(act),
                nn.AvgPool2d((1, 4)), nn.Dropout(0.4))
            self.block2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 15), padding=(0, 7),
                          groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2), make_act(act),
                nn.AvgPool2d((1, 8)), nn.Dropout(0.4))
            self.head = nn.Linear(F2 * (n_t // 32), n_cls)

        def forward(self, x):
            z = self.block2(self.block1(x))
            return self.head(z.flatten(1))

    return EEGNet()


def cv_subject(X, y, act, folds, seed=42):
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedKFold
    torch.set_num_threads(4)
    n_ch, n_t = X.shape[1], X.shape[2]
    # per-channel z-score (whole subject; scaling only)
    mu, sd = X.mean((0, 2), keepdims=True), X.std((0, 2), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    accs = []
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xn, y):
        torch.manual_seed(seed); np.random.seed(seed)
        net = build_cnn(n_ch, n_t, len(np.unique(y)), act)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        ce = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xn[tr]).unsqueeze(1); yt = torch.tensor(y[tr])
        idx = np.arange(len(tr))
        for ep in range(60):
            net.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), 32):
                bi = idx[b:b + 32]
                opt.zero_grad(); ce(net(Xt[bi]), yt[bi]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(Xn[te]).unsqueeze(1)).argmax(1).numpy()
        accs.append((pred == y[te]).mean())
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    subs = [f"sub-{i:03d}" for i in range(1, args.subjects + 1)]
    eeg = build_epochs(subs)
    print(f"=== MI activation sweep | eegmmidb left/right | {subs} | "
          f"{eeg.n_trials} trials, {eeg.n_channels}ch x {eeg.n_times} @ {SFREQ:g}Hz "
          f"| chance 0.50, subject-specific {args.folds}-fold ===\n", flush=True)

    subj_ids = sorted(set(eeg.subjects.tolist()))
    report = {}
    for act in ACTS:
        t0 = time.time()
        per_subj = []
        for s in subj_ids:
            m = eeg.subjects == s
            per_subj.append(cv_subject(eeg.X[m], eeg.y[m], act, args.folds))
        mean, std = float(np.mean(per_subj)), float(np.std(per_subj))
        report[act] = {"mean": mean, "std": std, "per_subject": per_subj}
        print(f"{act:12s} acc={mean:.3f} +/- {std:.3f}   "
              f"per-subj={[round(x,2) for x in per_subj]}   [{time.time()-t0:.0f}s]",
              flush=True)

    best = max(report, key=lambda a: report[a]["mean"])
    print(f"\nBEST activation: {best} ({report[best]['mean']:.3f})")
    out = ROOT / "results" / "metrics" / "mi_activation.json"
    out.write_text(json.dumps({"dataset": "eegmmidb_leftright", "subjects": subs,
                               "folds": args.folds, "chance": 0.5,
                               "results": report, "best": best}, indent=2),
                   encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

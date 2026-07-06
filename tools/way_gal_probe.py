#!/usr/bin/env python
"""Run our top-3 techniques on WAY-EEG-GAL (grasp-and-lift, 32-ch EEG @ 500 Hz).

Dataset: 12 participants x 328 grasp-and-lift trials, object weight (165/330/660 g)
and surface friction varied trial-to-trial. Per-trial windowed EEG lives in
WS_P*_S*.mat (ws.win(n).eeg = samples x 32, .eeg_t, .LEDon, .weight, .surf).

Tasks:
  * move-vs-rest : peri-LED movement window vs pre-LED baseline (chance 0.50,
                   POSITIVE CONTROL -- should be easily decodable).
  * weight       : 165 vs 330 vs 660 g from the execution window (chance 0.33,
                   the hard "intent parameter" decode).

Top-3 techniques (subject-specific K-fold + leave-one-series-out):
  1. Riemannian tangent + LDA
  2. Connectivity (PLV+imcoh+wPLI) -> PLS -> LDA
  3. MLP on tangent features (Adam; sweeps activation gelu/relu/elu/silu/...)

Usage: py tools/way_gal_probe.py --task move_rest --subjects P1 P2 P3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.next_experiments as nx
from src.config import load_config
from src.containers import TrialEpochs

DATA = ROOT / "data" / "way_eeg_gal"
FS = 500.0
WEIGHT_MAP = {1: 0, 2: 1, 4: 2}          # 165/330/660 g -> class 0/1/2


def _load_ws(path):
    from scipy.io import loadmat
    m = loadmat(path, struct_as_record=False, squeeze_me=True)
    return m["ws"]


def _crop(eeg, t, t0, t1):
    idx = np.where((t >= t0) & (t < t1))[0]
    return eeg[:, idx] if idx.size else None


def build_epochs(subjects, task, win_sec=1.5):
    """task in {move_rest, weight, surf}."""
    Xs, y, subs, runs = [], [], [], []
    win = int(round(win_sec * FS))
    for subj in subjects:
        files = sorted(glob.glob(str(DATA / "**" / f"WS_{subj}_S*.mat"),
                                 recursive=True))
        if not files:
            print(f"  [no WS files for {subj}]")
            continue
        for series, f in enumerate(files, 1):
            ws = _load_ws(f)
            wins = np.atleast_1d(ws.win)
            for w in wins:
                eeg = np.asarray(w.eeg, dtype=np.float64).T          # (32, T)
                t = np.asarray(w.eeg_t, dtype=np.float64)
                led = float(np.atleast_1d(w.LEDon)[0])
                if task == "move_rest":
                    mv = _crop(eeg, t, led, led + win_sec)
                    rs = _crop(eeg, t, led - win_sec, led)
                    for seg, lab in ((rs, 0), (mv, 1)):
                        if seg is not None and seg.shape[1] >= win:
                            Xs.append(seg[:, :win]); y.append(lab)
                            subs.append(subj); runs.append(series)
                else:
                    seg = _crop(eeg, t, led, led + win_sec)
                    if seg is None or seg.shape[1] < win:
                        continue
                    if task == "weight":
                        lab = WEIGHT_MAP.get(int(np.atleast_1d(w.weight)[0]))
                    else:  # surf
                        lab = int(np.atleast_1d(w.surf)[0]) - 1
                    if lab is None or lab < 0:
                        continue
                    Xs.append(seg[:, :win]); y.append(lab)
                    subs.append(subj); runs.append(series)
    if not Xs:
        raise RuntimeError("no trials built -- check extraction/paths")
    y = np.array(y); subs = np.array(subs); runs = np.array(runs)
    if task == "weight":
        # keep only weight-VARYING series (>=2 weight classes, each >=3 trials);
        # surface-varying series hold weight constant and would confound.
        keep = np.zeros(len(y), dtype=bool)
        for s in set(subs.tolist()):
            for sr in set(runs[subs == s].tolist()):
                m = (subs == s) & (runs == sr)
                vals, cnts = np.unique(y[m], return_counts=True)
                if (cnts >= 3).sum() >= 2:
                    keep |= m
        Xs = [x for x, k in zip(Xs, keep) if k]
        y, subs, runs = y[keep], subs[keep], runs[keep]
    X = np.stack(Xs, axis=0).astype(np.float32)
    classes = (["rest", "move"] if task == "move_rest" else
               ["165g", "330g", "660g"] if task == "weight" else
               ["sandpaper", "suede", "silk"])
    times = np.arange(win) / FS
    return TrialEpochs(X=X, y=np.array(y), sfreq=FS,
                       ch_names=[f"ch{i}" for i in range(X.shape[1])],
                       times=times, subjects=np.array(subs),
                       runs=np.array(runs),
                       uids=np.array([f"{s}_{i}" for i, s in enumerate(subs)]),
                       modality="eeg", classes=classes)


# ---- technique 3: configurable-activation MLP on tangent features ----------
def mlp_fold(sfreq, activation="gelu", optimizer="adamw", l_freq=8.0,
             h_freq=30.0, reg=1e-2, hidden=(128, 64), dropout=0.4, lr=1e-3,
             weight_decay=1e-3, epochs=150, patience=20, seed=42):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    from src.riemannian import RiemannianTangentSpace

    ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "elu": nn.ELU, "silu": nn.SiLU,
            "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}

    def build(d_in, n_cls):
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTS[activation](), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_cls)]
        return nn.Sequential(*layers)

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        torch.manual_seed(seed); np.random.seed(seed)
        ts = RiemannianTangentSpace(sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                    reg=reg).fit(eeg_tr, y_tr)
        sc = StandardScaler().fit(ts.transform(eeg_tr))
        Ztr = sc.transform(ts.transform(eeg_tr)).astype(np.float32)
        Zte = sc.transform(ts.transform(eeg_te)).astype(np.float32)
        classes = np.unique(y_tr)
        ye = np.searchsorted(classes, y_tr)
        # stratified val split
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(np.arange(len(ye)), test_size=0.2,
                                  stratify=ye, random_state=seed)
        net = build(Ztr.shape[1], len(classes))
        opt = ({"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
                "sgd": lambda p, **k: torch.optim.SGD(p, momentum=0.9, **k)}
               [optimizer])(net.parameters(), lr=lr, weight_decay=weight_decay)
        ce = nn.CrossEntropyLoss()
        Xt = torch.tensor(Ztr); yt = torch.tensor(ye, dtype=torch.long)
        best, best_loss, stale = None, 1e9, 0
        for ep in range(epochs):
            net.train(); opt.zero_grad()
            ce(net(Xt[tr]), yt[tr]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(ce(net(Xt[va]), yt[va]))
            if vl < best_loss - 1e-4:
                best_loss, stale = vl, 0
                best = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best:
            net.load_state_dict(best)
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(Zte)).argmax(1).numpy()
        return classes[pred]
    return fold


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["move_rest", "weight", "surf"],
                    default="move_rest")
    ap.add_argument("--subjects", nargs="*", default=["P1", "P2", "P3"])
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    help="tangent/conn band; default per task")
    args = ap.parse_args()

    lf, hf = (args.band if args.band else
              (8.0, 30.0) if args.task != "move_rest" else (1.0, 40.0))
    win_sec = 1.5 if args.task == "move_rest" else 2.5   # weight: capture liftoff
    eeg = build_epochs(args.subjects, args.task, win_sec=win_sec)
    nx.CLASSES = eeg.classes
    chance = 1.0 / len(eeg.classes)
    fn0 = np.zeros((eeg.n_trials, 0), dtype=np.float64)
    bands = [(lf, hf)] if args.task == "move_rest" else [(8, 13), (13, 30)]
    per = {c: int((eeg.y == i).sum()) for i, c in enumerate(eeg.classes)}
    print(f"=== WAY-EEG-GAL {args.task} | subjects={args.subjects} ===")
    print(f"{eeg.n_trials} trials, {eeg.n_channels}ch x {eeg.n_times} @ {FS:g}Hz "
          f"| classes(chance={chance:.2f})={per} | tangent band={lf}-{hf}Hz\n")

    cfg = load_config(None)
    report = {}

    def do(tag, fold):
        t0 = time.time()
        r = nx.run_within_subject(eeg, fn0, fold, cfg, seed=42)
        report[tag] = r
        s, l = r["subject"], r["loro"]
        print(f"{tag:<40} subj={s['mean_subject_acc']:.3f} "
              f"(bal={s['pooled_bal_acc']:.3f}) | LORO={l['mean_subject_acc']:.3f} "
              f"(bal={l['pooled_bal_acc']:.3f})   [{time.time()-t0:.0f}s]", flush=True)

    print("--- technique 1: Riemannian tangent + LDA ---")
    do("Riemannian tangent + LDA", nx.tangent_fold(l_freq=lf, h_freq=hf,
                                                    reg=1e-2, clf="lda", sfreq=FS))
    print("--- technique 2: Connectivity -> PLS -> LDA ---")
    do("Connectivity[plv+imcoh+wpli] -> PLS(8) -> LDA",
       nx.conn_pls_fold(FS, bands, ["plv", "imcoh", "wpli"], None, 8, "lda"))
    print("--- technique 3: MLP on tangent features (Adam; activation sweep) ---")
    for act in ("gelu", "relu", "elu", "silu", "leaky_relu", "tanh"):
        do(f"MLP tangent [{act}] AdamW",
           mlp_fold(FS, activation=act, optimizer="adamw", l_freq=lf, h_freq=hf))

    out = ROOT / "results" / "metrics" / f"way_gal_{args.task}.json"
    out.write_text(json.dumps({"task": args.task, "subjects": args.subjects,
                               "chance": chance, "per_class": per,
                               "results": report}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

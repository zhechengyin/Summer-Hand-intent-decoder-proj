#!/usr/bin/env python
"""Iteration 12: channel-selection SCORING analysis (no training, fast).

Answers the questions the live-selection design hinges on:
  1. Is the firing-rate top-8 STABLE across sessions? (if yes, adaptation is
     insurance not accuracy; if it drifts, adaptation is justified.)
  2. Does firing rate already pick the VELOCITY-RELEVANT channels, or is there
     room for a better score? (overlap of firing-top8 with velocity-corr top8.)
  3. Do the proposed scores agree or diverge? firing / low-freq power (0.2-3 Hz) /
     velocity-corr / FFT-weighted (sum f*P).

Per-channel scores on 96 electrodes, per session, then compared. 25 Hz rates.
Usage: py research/iter12_channel_scores.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.iter7_final as I7
import models.tcn_gru.evaluate as E

FS = 1.0 / E.BIN                    # 25 Hz
TRAIN = list(E.TRAIN) + I7.EXTRA18  # 24 sessions
K = 8


def scores(rates, vel):
    """Per-channel (96,) scores. rates (96,T), vel (T,2)."""
    firing = rates.mean(1)
    f, P = welch(rates, fs=FS, nperseg=min(256, rates.shape[1]), axis=1)  # (96, nf)
    lo = (f >= 0.2) & (f <= 3.0)
    lowfreq = P[:, lo].sum(1)                                   # 0.2-3 Hz power
    fftw = (f * P).sum(1)                                       # frequency-weighted
    vx, vy = vel[:, 0], vel[:, 1]
    rc = rates - rates.mean(1, keepdims=True)
    cx = (rc * (vx - vx.mean())).sum(1) / (np.linalg.norm(rc, axis=1) * np.linalg.norm(vx - vx.mean()) + 1e-9)
    cy = (rc * (vy - vy.mean())).sum(1) / (np.linalg.norm(rc, axis=1) * np.linalg.norm(vy - vy.mean()) + 1e-9)
    velcorr = np.sqrt(cx ** 2 + cy ** 2)                       # LOG-047 corr score
    return {"firing": firing, "lowfreq": lowfreq, "velcorr": velcorr, "fftweighted": fftw}


def top(s):
    return set(np.argsort(s)[-K:].tolist())


def overlap(a, b):
    return len(a & b) / K


def main():
    per = {s: scores(*E.load_electrode(s)) for s in TRAIN}      # {session: {score: (96,)}}
    keys = ["firing", "lowfreq", "velcorr", "fftweighted"]
    glob = {k: np.mean([per[s][k] for s in TRAIN], 0) for k in keys}   # avg over train
    gtop = {k: top(glob[k]) for k in keys}

    print("=== Channel-selection scoring analysis (24 sessions, 96 ch) ===\n")
    print("1) STABILITY of each score's top-8 across the 24 sessions")
    print("   (mean pairwise overlap of per-session top-8; 1.0 = identical every session)")
    stab = {}
    for k in keys:
        tops = [top(per[s][k]) for s in TRAIN]
        ov = np.mean([overlap(tops[i], tops[j])
                      for i in range(len(tops)) for j in range(i + 1, len(tops))])
        # how often each globally-chosen channel is in a session's own top-8
        hit = np.mean([len(gtop[k] & t) / K for t in tops])
        stab[k] = {"pairwise_overlap": float(ov), "global_hit_rate": float(hit)}
        print(f"   {k:12s}: pairwise {ov:.2f}   global-top8 present {hit:.2f} of sessions")

    print("\n2) Does FIRING rate pick the VELOCITY-relevant channels?")
    print(f"   firing-top8 vs velcorr-top8 overlap: {overlap(gtop['firing'], gtop['velcorr']):.2f}")
    print(f"   firing-top8 = {sorted(gtop['firing'])}")
    print(f"   velcorr-top8 = {sorted(gtop['velcorr'])}")

    print("\n3) Agreement between scores (global top-8 overlap vs firing)")
    for k in ("lowfreq", "velcorr", "fftweighted"):
        print(f"   firing vs {k:12s}: {overlap(gtop['firing'], gtop[k]):.2f}")

    out = ROOT / "results" / "metrics" / "iter12_channel_scores.json"
    out.write_text(json.dumps({"stability": stab,
                               "global_top8": {k: sorted(gtop[k]) for k in keys},
                               "firing_vs": {k: overlap(gtop["firing"], gtop[k])
                                             for k in keys}}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

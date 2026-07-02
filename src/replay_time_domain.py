"""Stage 8 -- time-domain replay / online-style demo.

Replays dataset trials as if biosignal windows were arriving in real time and
drives the N1 -> N2 -> avatar loop:

    for each time step t:
        x_t     = next signal window
        p_t     = N1(x_t)                      # probability vector
        s_t     = current prosthetic state
        u_t     = N2(p_t, s_t)                 # state-aware command
        s_{t+1} = simulate(u_t)                # avatar updates

To make it genuinely time-domain (not a single static prediction per trial), each
trial is sliced into several sub-windows; N1 emits a probability vector per
window, so N2 sees an evolving stream and its smoothing/gating actually matters.

A held-out train/replay split (by trial) ensures the replayed trials were not used
to fit the streaming N1.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from .config import cfg_get, resolve_path
from .containers import TrialEpochs
from .feature_extraction import _bandpower
from .fusion import FeatureSet
from .mini_ai_spine_n2 import N2Interpreter
from .simulate_avatar import AvatarSimulator
from .state import ProstheticState
from .train_n1 import N1Decoder


@dataclass
class TrialStream:
    uid: str
    subject: str
    run: int
    label: int
    feats: np.ndarray          # (n_windows, n_features)
    win_times: list[float]     # window-centre time (s) relative to onset


# ---------------------------------------------------------------------------
# Window-level (streaming) features -- one feature vector per sub-window
# ---------------------------------------------------------------------------
def window_features_per_trial(epochs: TrialEpochs, cfg: dict, n_windows: int):
    """Per trial, return a (n_windows, n_features) matrix of per-window bandpower.

    Unlike feature_extraction.eeg_features (which concatenates all windows into
    one vector), here each window is its own feature row so it can be streamed.
    """
    bands = cfg_get(cfg, "features.eeg_bands", {"mu": [8, 13], "beta": [13, 30]})
    log_power = bool(cfg_get(cfg, "features.eeg_log_power", True))
    fs = epochs.sfreq
    # split the epoch into n_windows contiguous windows
    edges = np.linspace(0, epochs.n_times, n_windows + 1).astype(int)
    bounds = [(edges[i], edges[i + 1]) for i in range(n_windows)]
    band_items = list(bands.items())

    names = [f"{ch}|{bn}" for ch in epochs.ch_names for bn, _ in band_items]
    streams: list[TrialStream] = []
    for t in range(epochs.n_trials):
        feats = np.empty((n_windows, epochs.n_channels * len(band_items)),
                         dtype=np.float32)
        wtimes = []
        for wi, (a, b) in enumerate(bounds):
            seg = epochs.X[t, :, a:b]
            wtimes.append(float(epochs.times[(a + b) // 2]))
            col = 0
            for ch_i in range(epochs.n_channels):
                for _, brange in band_items:
                    bp = _bandpower(seg[ch_i], fs, tuple(brange))
                    feats[wi, col] = np.log(bp + 1e-12) if log_power else bp
                    col += 1
        streams.append(TrialStream(
            uid=str(epochs.uids[t]), subject=str(epochs.subjects[t]),
            run=int(epochs.runs[t]), label=int(epochs.y[t]),
            feats=feats, win_times=wtimes))
    return streams, names


def _streams_to_featureset(streams, names, classes) -> FeatureSet:
    X = np.vstack([s.feats for s in streams])
    y = np.concatenate([[s.label] * s.feats.shape[0] for s in streams])
    subs = np.concatenate([[s.subject] * s.feats.shape[0] for s in streams])
    runs = np.concatenate([[s.run] * s.feats.shape[0] for s in streams])
    uids = np.concatenate([[f"{s.uid}|w{w}" for w in range(s.feats.shape[0])]
                           for s in streams])
    return FeatureSet(X.astype(np.float32), list(names), y, subs, runs, uids,
                      "eeg_windows", list(classes))


def _fmt_proba(p: dict) -> str:
    return " ".join(f"{k}={v:.2f}" for k, v in p.items())


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------
def run_replay(cfg: dict, eeg_epochs: TrialEpochs, animate: bool | None = None,
               max_trials: int = 8, seed: int | None = None) -> list[dict]:
    n_windows = int(cfg_get(cfg, "replay.windows_per_trial", 5))
    if animate is None:
        animate = bool(cfg_get(cfg, "replay.animate", False))
    seed = int(cfg_get(cfg, "seed", 42)) if seed is None else seed
    classes = eeg_epochs.classes

    streams, names = window_features_per_trial(eeg_epochs, cfg, n_windows)

    # held-out split by trial so replayed trials are unseen by the streaming N1
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(streams))
    n_train = max(1, int(0.7 * len(streams)))
    train_idx, test_idx = order[:n_train], order[n_train:]
    n1 = N1Decoder.train(
        _streams_to_featureset([streams[i] for i in train_idx], names, classes),
        cfg)

    n2 = N2Interpreter(cfg)
    state = ProstheticState()
    avatar = AvatarSimulator()
    replay_trials = [streams[i] for i in test_idx[:max_trials]]

    log: list[dict] = []
    frames: list[dict] = []
    step = 0
    print("\n" + "=" * 96)
    print("TIME-DOMAIN REPLAY  (N1 -> N2 -> avatar).  true = ground-truth imagined action")
    print("=" * 96)
    for tr in replay_trials:
        print(f"\n--- trial {tr.uid}  true={classes[tr.label].upper()} ---")
        for wi in range(tr.feats.shape[0]):
            out = n1.predict_one(tr.feats[wi])
            res = n2.step(out.probabilities, state)
            state = res.next_state
            step += 1
            row = {
                "step": step, "trial": tr.uid, "t_sec": round(tr.win_times[wi], 2),
                "true": classes[tr.label], "n1_intent": out.intent,
                "n1_confidence": round(out.confidence, 3),
                "n1_probs": {k: round(v, 3) for k, v in out.probabilities.items()},
                "n2_command": res.prosthetic_action, "accepted": res.accepted,
                "reason": res.reason, "state": state.summary(),
            }
            log.append(row)
            frames.append({"proba": out.probabilities, "state": state.copy(),
                           "true": classes[tr.label], "cmd": res.prosthetic_action,
                           "intent": out.intent, "t": tr.win_times[wi]})
            hit = "OK " if out.intent == classes[tr.label] else "  x"
            print(f"  t={row['t_sec']:>4.1f}s {hit} N1[{_fmt_proba(row['n1_probs'])}] "
                  f"-> {out.intent:<6} conf={out.confidence:.2f} | "
                  f"N2={res.prosthetic_action:<20} {avatar.render_ascii(state)}")

    _write_log_csv(cfg, log)
    _summary(log, classes)
    if animate:
        _animate(cfg, frames, classes)
    return log


def _write_log_csv(cfg: dict, log: list[dict]) -> None:
    path = resolve_path(cfg, "paths.metrics_dir") / "replay_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "trial", "t_sec", "true", "n1_intent",
                    "n1_confidence", "n2_command", "accepted", "reason"])
        for r in log:
            w.writerow([r["step"], r["trial"], r["t_sec"], r["true"],
                        r["n1_intent"], r["n1_confidence"], r["n2_command"],
                        r["accepted"], r["reason"]])
    print(f"\nReplay log -> {path}")


def _summary(log: list[dict], classes) -> None:
    if not log:
        return
    n = len(log)
    win_hits = sum(r["n1_intent"] == r["true"] for r in log)
    accepted = sum(r["accepted"] for r in log)
    print(f"\nReplay summary: {n} windows | N1 window-accuracy="
          f"{win_hits / n:.2f} (chance={1/len(classes):.2f}) | "
          f"N2 accepted {accepted}/{n} commands "
          f"({100*accepted/n:.0f}%), deferred {n-accepted}.")


def _animate(cfg: dict, frames: list[dict], classes) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:  # pragma: no cover
        print(f"[replay] animation unavailable ({e}); CLI output only.")
        return

    avatar = AvatarSimulator()
    fig, (ax_arm, ax_bar) = plt.subplots(1, 2, figsize=(9, 4.2))

    def render(i):
        f = frames[i]
        avatar.draw(ax_arm, f["state"],
                    title=f"t={f['t']:.1f}s  true={f['true'].upper()}  "
                          f"cmd={f['cmd']}")
        ax_bar.clear()
        keys = list(f["proba"].keys())
        vals = [f["proba"][k] for k in keys]
        colors = ["#5b8def" if k == f["intent"] else "#c9d6ea" for k in keys]
        ax_bar.bar(keys, vals, color=colors)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("N1 probability")
        ax_bar.set_title(f"N1 intent = {f['intent'].upper()}")
        ax_bar.axhline(1 / len(classes), ls="--", lw=0.8, color="#999")

    anim = FuncAnimation(fig, render, frames=len(frames), interval=500)
    out = resolve_path(cfg, "paths.figures_dir") / "replay.gif"
    try:
        anim.save(out, writer=PillowWriter(fps=2))
        print(f"Replay animation -> {out}")
    except Exception as e:  # pragma: no cover
        print(f"[replay] could not save gif ({e}).")
    plt.close(fig)

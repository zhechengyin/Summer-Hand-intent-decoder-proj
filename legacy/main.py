#!/usr/bin/env python
"""Neural Intent Decoder -- command-line orchestrator.

Stages wire together into subcommands:

    python main.py inspect     # Stage 1: discover files, print events/channels
    python main.py preprocess  # Stages 2: build + cache EEG/fNIRS epochs
    python main.py train       # Stage 5: fit N1 on a modality, save the model
    python main.py evaluate    # Stages 3-7: features, fusion, CV, metrics, figs
    python main.py fbcsp       # EEG FBCSP + fNIRS evaluation
    python main.py riemannian  # EEG tangent-space + fNIRS evaluation
    python main.py temporal-cnn # EEG temporal CNN + fNIRS evaluation
    python main.py smoothed-temporal-cnn # 10-sample smoothed EEG CNN
    python main.py symbolic-erd # symbolic ERD/ERS timing + fNIRS evaluation
    python main.py snn         # bare EEG SNN + fNIRS evaluation
    python main.py riemannian-snn # tangent-space SNN + fNIRS evaluation
    python main.py windowed-riemannian-snn # windowed tangent-space SNN
    python main.py demo        # Stage 8: time-domain N1->N2->avatar replay
    python main.py smoke       # run the WHOLE pipeline on synthetic data (no
                               #          dataset download needed) -- great for a
                               #          first end-to-end check
    python main.py all         # preprocess -> evaluate -> demo on the real data

Common options: --config, --classifier {lda,logreg,svm,rf,gb,gelu_nn}, --subjects,
--no-cache, --animate.
"""
from __future__ import annotations

import argparse
import sys

from src.config import cfg_get, ensure_dirs, load_config, resolve_path, seed_everything


def _load(cfg_path):
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    seed_everything(int(cfg_get(cfg, "seed", 42)))
    return cfg


def _apply_overrides(cfg, args):
    if getattr(args, "classifier", None):
        cfg["model"]["classifier"] = args.classifier
    if getattr(args, "animate", False):
        cfg.setdefault("replay", {})["animate"] = True
    return cfg


# ---------------------------------------------------------------------------
# real-data epoch loading
# ---------------------------------------------------------------------------
def _build_epochs(cfg, subjects, use_cache):
    from src.load_bids import discover_dataset
    from src.preprocess_eeg import build_eeg_epochs
    from src.preprocess_fnirs import build_fnirs_epochs

    root = resolve_path(cfg, "paths.bids_root")
    if not root.exists():
        print(f"Dataset not found at {root}.\n"
              "Download ds004022 first (see data/README.md) or run "
              "`python main.py smoke` to try the pipeline on synthetic data.",
              file=sys.stderr)
        return None, None
    index = discover_dataset(root)
    if not index.runs:
        print(f"No BIDS runs discovered under {root}.", file=sys.stderr)
        return None, None
    eeg = build_eeg_epochs(cfg, index, subjects, cache=use_cache)
    fnirs = build_fnirs_epochs(cfg, index, subjects, cache=use_cache)
    return eeg, fnirs


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def cmd_inspect(cfg, args):
    from src.load_bids import inspect_dataset

    inspect_dataset(cfg, max_runs=args.max_runs)


def cmd_preprocess(cfg, args):
    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is not None:
        print(eeg.summary())
    if fnirs is not None:
        print(fnirs.summary())


def cmd_train(cfg, args):
    from src.fusion import build_feature_sets
    from src.train_n1 import N1Decoder

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    sets = build_feature_sets(cfg, eeg, fnirs)
    fs = sets.get("fused")
    if fs is None:
        print("No fused EEG+fNIRS feature set available.", file=sys.stderr)
        return
    n1 = N1Decoder.train(fs, cfg)
    out = resolve_path(cfg, "paths.results_dir") / "n1_fused.joblib"
    n1.save(out)
    print(f"Trained N1 (fused EEG+fNIRS, {cfg_get(cfg,'model.classifier')}) -> {out}")


def cmd_evaluate(cfg, args):
    from src.fusion import build_feature_sets
    from src.evaluate import evaluate_all

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None and fnirs is None:
        return
    sets = build_feature_sets(cfg, eeg, fnirs)
    if "fused" not in sets:
        print("No fused EEG+fNIRS feature set available.", file=sys.stderr)
        return
    evaluate_all({"fused": sets["fused"]}, cfg, loro=not args.no_loro)


def cmd_demo(cfg, args):
    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("Demo now requires both EEG and fNIRS epochs.", file=sys.stderr)
        return
    from src.replay_time_domain import run_replay

    run_replay(cfg, eeg, fnirs, animate=args.animate, max_trials=args.max_trials)


def cmd_smoke(cfg, args):
    """Whole pipeline on synthetic data -- no dataset required."""
    from src.evaluate import evaluate_all
    from src.fusion import build_feature_sets
    from src.replay_time_domain import run_replay
    from src.synthetic import make_synthetic

    print(">> Generating synthetic EEG + fNIRS epochs ...")
    eeg, fnirs = make_synthetic(cfg, n_subjects=3, n_runs=2, per_class=8)
    print("  " + eeg.summary())
    print("  " + fnirs.summary())

    print("\n>> Building features and evaluating fused EEG+fNIRS ...")
    sets = build_feature_sets(cfg, eeg, fnirs)
    evaluate_all({"fused": sets["fused"]}, cfg, loro=not args.no_loro)

    print("\n>> Running the time-domain N1->N2->avatar replay ...")
    run_replay(cfg, eeg, fnirs, animate=args.animate, max_trials=args.max_trials)
    print("\nSmoke test complete.")


def cmd_all(cfg, args):
    from src.evaluate import evaluate_all
    from src.fusion import build_feature_sets
    from src.replay_time_domain import run_replay

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None and fnirs is None:
        return
    sets = build_feature_sets(cfg, eeg, fnirs)
    if "fused" not in sets:
        print("No fused EEG+fNIRS feature set available.", file=sys.stderr)
        return
    evaluate_all({"fused": sets["fused"]}, cfg, loro=not args.no_loro)
    if eeg is not None and fnirs is not None:
        run_replay(cfg, eeg, fnirs, animate=args.animate, max_trials=args.max_trials)


def cmd_sweep(cfg, args):
    """Sweep analysis windows to find the best EEG/fNIRS decoding window."""
    from src.window_sweep import run_sweep

    root = resolve_path(cfg, "paths.bids_root")
    if not root.exists():
        print(f"Dataset not found at {root} (see data/README.md).", file=sys.stderr)
        return
    run_sweep(cfg, args.modality, subjects=args.subjects, name=args.classifier)


def cmd_fbcsp(cfg, args):
    """Evaluate FBCSP EEG features fused with fNIRS."""
    from src.evaluate import evaluate_fbcsp

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("FBCSP now requires both EEG and fNIRS epochs.", file=sys.stderr)
        return
    evaluate_fbcsp(eeg, fnirs, cfg, name=args.classifier,
                   loro=not args.no_loro)


def cmd_riemannian(cfg, args):
    """Evaluate Riemannian EEG features fused with fNIRS."""
    from src.evaluate import evaluate_riemannian

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("Riemannian evaluation now requires both EEG and fNIRS epochs.",
              file=sys.stderr)
        return
    evaluate_riemannian(eeg, fnirs, cfg, name=args.classifier,
                        loro=not args.no_loro)


def cmd_temporal_cnn(cfg, args):
    """Evaluate raw-EEG temporal CNN fused with fNIRS."""
    from src.evaluate import evaluate_temporal_cnn

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("Temporal CNN evaluation requires both EEG and fNIRS epochs.",
              file=sys.stderr)
        return
    evaluate_temporal_cnn(eeg, fnirs, cfg, loro=not args.no_loro)


def cmd_smoothed_temporal_cnn(cfg, args):
    """Evaluate 10-sample moving-average EEG temporal CNN fused with fNIRS."""
    from src.evaluate import evaluate_temporal_cnn

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("Smoothed temporal CNN evaluation requires EEG and fNIRS epochs.",
              file=sys.stderr)
        return
    evaluate_temporal_cnn(
        eeg, fnirs, cfg, loro=not args.no_loro,
        prefix="smoothed_temporal_cnn",
        stem="smoothed_temporal_cnn",
        label="10-sample moving-average temporal CNN")


def cmd_symbolic_erd(cfg, args):
    """Evaluate symbolic EEG ERD/ERS timing features fused with fNIRS."""
    from src.evaluate import evaluate_symbolic_erd
    from src.load_bids import discover_dataset
    from src.preprocess_eeg import build_eeg_epochs
    from src.preprocess_fnirs import build_fnirs_epochs

    root = resolve_path(cfg, "paths.bids_root")
    if not root.exists():
        print(f"Dataset not found at {root}.", file=sys.stderr)
        return
    index = discover_dataset(root)
    tmin = float(cfg_get(cfg, "symbolic_erd.eeg_tmin", -2.0))
    tmax = float(cfg_get(cfg, "symbolic_erd.eeg_tmax", 5.0))
    eeg = build_eeg_epochs(cfg, index, args.subjects, cache=False,
                           tmin=tmin, tmax=tmax)
    fnirs = build_fnirs_epochs(cfg, index, args.subjects,
                               cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("Symbolic ERD evaluation requires EEG and fNIRS epochs.",
              file=sys.stderr)
        return
    evaluate_symbolic_erd(eeg, fnirs, cfg, name=args.classifier,
                          loro=not args.no_loro)


def cmd_snn(cfg, args):
    """Evaluate bare SNN fused with fNIRS."""
    from src.evaluate import evaluate_snn

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("SNN evaluation requires both EEG and fNIRS epochs.",
              file=sys.stderr)
        return
    evaluate_snn(eeg, fnirs, cfg, loro=not args.no_loro)


def cmd_riemannian_snn(cfg, args):
    """Evaluate Riemannian tangent-space features with an SNN head."""
    from src.evaluate import evaluate_riemannian_snn

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("Riemannian SNN evaluation requires both EEG and fNIRS epochs.",
              file=sys.stderr)
        return
    evaluate_riemannian_snn(eeg, fnirs, cfg, loro=not args.no_loro,
                            windowed=False)


def cmd_windowed_riemannian_snn(cfg, args):
    """Evaluate windowed Riemannian tangent-space features with an SNN head."""
    from src.evaluate import evaluate_riemannian_snn

    eeg, fnirs = _build_epochs(cfg, args.subjects, use_cache=not args.no_cache)
    if eeg is None or fnirs is None:
        print("Windowed Riemannian SNN evaluation requires EEG and fNIRS epochs.",
              file=sys.stderr)
        return
    evaluate_riemannian_snn(eeg, fnirs, cfg, loro=not args.no_loro,
                            windowed=True)


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--subjects", nargs="*", default=None,
                        help="restrict to e.g. sub-01 sub-02")
        sp.add_argument("--classifier", default=None,
                        choices=["lda", "logreg", "svm", "rf", "gb", "xgb",
                                 "gelu_nn"])
        sp.add_argument("--no-cache", action="store_true",
                        help="ignore cached epochs and rebuild")
        sp.add_argument("--no-loro", action="store_true",
                        help="skip leave-one-run-out")
        sp.add_argument("--animate", action="store_true",
                        help="render the replay as a gif")
        sp.add_argument("--max-trials", type=int, default=8,
                        help="trials to replay in the demo")

    sp = sub.add_parser("inspect"); sp.add_argument("--max-runs", type=int, default=2)
    for name in ("preprocess", "evaluate", "demo", "smoke", "all", "fbcsp",
                 "riemannian", "temporal-cnn", "smoothed-temporal-cnn",
                 "symbolic-erd", "snn", "riemannian-snn",
                 "windowed-riemannian-snn"):
        common(sub.add_parser(name))
    tr = sub.add_parser("train"); common(tr)
    sw = sub.add_parser("sweep"); common(sw)
    sw.add_argument("--modality", default="eeg", choices=["eeg", "fnirs", "both"])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = _apply_overrides(_load(args.config), args)
    dispatch = {
        "inspect": cmd_inspect, "preprocess": cmd_preprocess, "train": cmd_train,
        "evaluate": cmd_evaluate, "demo": cmd_demo, "smoke": cmd_smoke,
        "all": cmd_all, "sweep": cmd_sweep, "fbcsp": cmd_fbcsp,
        "riemannian": cmd_riemannian, "temporal-cnn": cmd_temporal_cnn,
        "smoothed-temporal-cnn": cmd_smoothed_temporal_cnn,
        "symbolic-erd": cmd_symbolic_erd,
        "snn": cmd_snn, "riemannian-snn": cmd_riemannian_snn,
        "windowed-riemannian-snn": cmd_windowed_riemannian_snn,
    }
    dispatch[args.command](cfg, args)


if __name__ == "__main__":
    main()

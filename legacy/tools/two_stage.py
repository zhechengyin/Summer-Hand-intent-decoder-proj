#!/usr/bin/env python
"""Two-stage 'identify-then-decode' system.

Motivation: Riemannian tangent features identify the SUBJECT almost perfectly
(subject-ID ~0.99 in our probes) while only weakly reading the class. So use a
router:

    Stage 1 (profile detection): Riemannian tangent -> classifier -> subject id
    Stage 2 (decode):            apply THAT subject's own class decoder

This targets the "unknown identity at test time" scenario. Evaluated with a
cohort-level leave-one-run-out split (train on other runs of ALL subjects, test
on the held-out run) so nothing leaks. We compare:

    pooled     : one class decoder trained on everyone (ignores identity)
    two-stage  : predicted-subject routing (this system)
    oracle     : true-subject routing (upper bound == per-subject decoder)

Usage: py tools/two_stage.py --dataset {eegmmidb,cursor,ds004022}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.riemannian import RiemannianTangentSpace


def _tangent(sf, l, h, reg):
    return RiemannianTangentSpace(sfreq=sf, l_freq=l, h_freq=h, reg=reg)


def run_two_stage(eeg, l_freq, h_freq, reg):
    X, y, subj, runs, sf = (eeg.X, eeg.y, eeg.subjects, eeg.runs, eeg.sfreq)
    subjects = sorted(set(subj.tolist()))
    sid = {s: i for i, s in enumerate(subjects)}
    id_true, id_pred = [], []
    ct, c_pool, c_two, c_oracle = [], [], [], []

    for r in sorted(set(runs.tolist())):
        tr, te = runs != r, runs == r
        if tr.sum() == 0 or te.sum() == 0:
            continue

        # Stage 1: subject-id from tangent (reference fit on all training trials)
        ts_all = _tangent(sf, l_freq, h_freq, reg).fit(X[tr])
        Ztr, Zte = ts_all.transform(X[tr]), ts_all.transform(X[te])
        sc = StandardScaler().fit(Ztr)
        Ztr_s, Zte_s = sc.transform(Ztr), sc.transform(Zte)
        s_clf = LogisticRegression(max_iter=2000).fit(
            Ztr_s, [sid[s] for s in subj[tr]])
        pred_id = s_clf.predict(Zte_s)

        # pooled class decoder (shared tangent geometry)
        pool = Pipeline([("sc", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis())]).fit(Ztr, y[tr])
        pool_pred = pool.predict(Zte)

        # per-subject class decoders (each fits its own tangent reference)
        submodels = {}
        for s in subjects:
            m = tr & (subj == s)
            if m.sum() < 4 or len(set(y[m].tolist())) < 2:
                submodels[s] = None
                continue
            ts_s = _tangent(sf, l_freq, h_freq, reg).fit(X[m])
            clf = Pipeline([("sc", StandardScaler()),
                            ("clf", LinearDiscriminantAnalysis())]).fit(
                ts_s.transform(X[m]), y[m])
            submodels[s] = (ts_s, clf)

        Xte, yte, ste = X[te], y[te], subj[te]

        def batch_predict(route_subj):
            out = np.array(pool_pred, copy=True)   # fallback = pooled
            for s in subjects:
                idx = np.where(route_subj == sid[s])[0]
                if len(idx) == 0 or submodels[s] is None:
                    continue
                ts_s, clf = submodels[s]
                out[idx] = clf.predict(ts_s.transform(Xte[idx]))
            return out

        oracle_route = np.array([sid[s] for s in ste])
        c_oracle.extend(batch_predict(oracle_route).tolist())
        c_two.extend(batch_predict(pred_id).tolist())
        c_pool.extend(pool_pred.tolist())
        ct.extend(yte.tolist())
        id_true.extend([sid[s] for s in ste])
        id_pred.extend(pred_id.tolist())

    ct = np.array(ct)
    def acc(p):
        return {"acc": float(accuracy_score(ct, p)),
                "bal_acc": float(balanced_accuracy_score(ct, p))}
    return {
        "subject_id_acc": float(accuracy_score(id_true, id_pred)),
        "pooled": acc(np.array(c_pool)),
        "two_stage": acc(np.array(c_two)),
        "oracle": acc(np.array(c_oracle)),
        "n_subjects": len(subjects), "n_test": int(len(ct)),
    }


def load_dataset(name, cfg):
    if name == "eegmmidb":
        from tools.eegmmidb_probe import build_epochs
        eeg = build_epochs([f"sub-{i:03d}" for i in range(1, 11)])
        return eeg, 8.0, 30.0, 1e-3, 0.5, "ds004362 left/right MI (10 subj)"
    if name == "cursor":
        from tools.cursor_probe import build_epochs
        return build_epochs(), 1.0, 40.0, 1e-2, 0.25, "cursor 4-class (4 subj)"
    if name == "ds004022":
        import tools.next_experiments as nx
        eeg, _ = nx.load_aligned(cfg)
        return eeg, 8.0, 30.0, 1e-3, 0.25, "ds004022 same-limb (7 subj, EEG-only)"
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["eegmmidb", "cursor", "ds004022"],
                    default="cursor")
    args = ap.parse_args()
    cfg = load_config(None)
    eeg, lf, hf, reg, chance, label = load_dataset(args.dataset, cfg)
    print(f"=== Two-stage identify-then-decode: {label} ===")
    print(f"{eeg.n_trials} trials, {eeg.n_channels}ch @ {eeg.sfreq:.0f}Hz, "
          f"chance={chance}\n")
    r = run_two_stage(eeg, lf, hf, reg)
    print(f"Stage-1 subject-ID accuracy : {r['subject_id_acc']:.3f} "
          f"({r['n_subjects']} subjects)")
    print(f"Class decode (chance {chance}):")
    print(f"  pooled  (ignore identity)  : {r['pooled']['acc']:.3f} "
          f"(bal {r['pooled']['bal_acc']:.3f})")
    print(f"  two-stage (predicted route): {r['two_stage']['acc']:.3f} "
          f"(bal {r['two_stage']['bal_acc']:.3f})")
    print(f"  oracle  (true route, ceil) : {r['oracle']['acc']:.3f} "
          f"(bal {r['oracle']['bal_acc']:.3f})")
    out = ROOT / "results" / "metrics" / f"two_stage_{args.dataset}.json"
    out.write_text(json.dumps({"dataset": args.dataset, "chance": chance,
                               "results": r}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

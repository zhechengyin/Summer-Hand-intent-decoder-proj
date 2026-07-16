"""Historical compatibility harness for archived experiments only.

This module deliberately keeps the old evaluator/model available to reproduce
past experiments, including non-causal variants. Active code must import
``src.intent_decoder.training`` instead.
"""
from __future__ import annotations

import numpy as np

import models.tcn_gru.best_model as legacy_model
import models.tcn_gru.evaluate as legacy_data
from src.intent_decoder.training import run as _run


def prep(nch=96):
    """Prepare the historical fixed split with its original methodology."""
    loaded = {
        session: legacy_data.load_electrode(session)
        for session in legacy_data.TRAIN + legacy_data.EVAL + legacy_data.TEST
    }
    variance = np.mean(
        [loaded[session][1].std(0) for session in legacy_data.TRAIN], axis=0
    )
    axes = np.sort(np.argsort(variance)[-2:])
    if nch < 96:
        firing = np.mean(
            [loaded[session][0].mean(1) for session in legacy_data.TRAIN], axis=0
        )
        selected = np.sort(np.argsort(firing)[-nch:])
    else:
        selected = np.arange(96)
    train = [
        trial
        for session in legacy_data.TRAIN
        for trial in legacy_data.windows(*loaded[session], axes)
    ]
    evaluation = {
        session: legacy_data.windows(*loaded[session], axes)
        for session in legacy_data.EVAL
    }
    test = {
        session: legacy_data.windows(*loaded[session], axes)
        for session in legacy_data.TEST
    }
    if nch < 96:
        # Rebuild after channel selection to match the original helper behavior.
        train = [
            trial
            for session in legacy_data.TRAIN
            for trial in legacy_data.windows(loaded[session][0][selected], loaded[session][1], axes)
        ]
        evaluation = {
            session: legacy_data.windows(loaded[session][0][selected], loaded[session][1], axes)
            for session in legacy_data.EVAL
        }
        test = {
            session: legacy_data.windows(loaded[session][0][selected], loaded[session][1], axes)
            for session in legacy_data.TEST
        }
    return {
        "train": train,
        "eval": evaluation,
        "test": test,
        "axes": axes,
        "sel": selected,
        "nch": len(selected),
    }


def run(data, cfg, build=None, **kwargs):
    """Run with the old future-capable architecture only inside the archive."""
    return _run(data, cfg, build=build or legacy_model.build_net, **kwargs)


__all__ = ["prep", "run"]

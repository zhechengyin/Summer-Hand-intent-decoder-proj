#!/usr/bin/env python
"""Reusable identity-preserving masked input for a dynamic-channel decoder (LOG-047 / RMTBD brief).

Motivation: LOG-083 showed per-session channel RE-SELECTION helps (~70% of the best-8 channels
churn over months), BUT feeding re-selected electrodes into a model whose input slots were trained
to mean SPECIFIC old electrodes hurts (resel_ft < resel_scratch) -- the model depends on electrode
IDENTITY. The fix: a fixed 96-electrode identity layout + an explicit observation mask, so the
active 32 (or 8) can change WITHOUT retraining.

DESIGN CHOICES (documented per the brief):
  * Layout: slot i == physical electrode i, ALWAYS. Selected electrodes are NEVER compacted into
    positions 0..N-1. An unselected electrode's slot is forced to exactly 0.
  * Neural features: 96 electrodes x {raw counts, causal EWMA} = 192, z-scored per session, then
    masked electrodes zeroed AFTER normalization (so normalization can never turn a missing input
    into nonzero evidence).
  * Mask encoding: the 96-dim binary observation mask is CONCATENATED as extra input channels
    (192 neural + 96 mask = 288). Chosen over gating for simplicity and so the 1x1 spatial-mix conv
    can learn to use identity + observed/unobserved jointly. The mask lets the model distinguish an
    UNOBSERVED electrode from an OBSERVED electrode that happens to have zero spikes.
  * BatchNorm sits AFTER the 1x1 conv (on F feature maps), never on the 192 neural inputs, so a
    masked (zero) electrode contributes exactly 0 to the conv output regardless of BN.

The reusable pieces here are deliberately framework-light: build_neural_192 (numpy) and
apply_mask_torch / make_masked_input_torch (torch, batch-time). A model is just
best_model.build_net(cfg, n_ch=288).
"""
from __future__ import annotations

import numpy as np

import experiments.archive.indy.iter25_causal_smoothing as I25

N_ELEC = 96


def build_neural_192(counts96, ewma_alpha=0.1):
    """(96, T) raw counts -> (192, T) z-scored [raw96 ; ewma96]. NOT masked yet.

    z-score is per-electrode over the whole array (label-free). Order: [raw block, ewma block],
    each block indexed by physical electrode 0..95."""
    c = counts96.astype(np.float32)
    feat = np.concatenate([c, I25.ewma(c, ewma_alpha)], 0)          # (192, T)
    mu = feat.mean(1, keepdims=True)
    sd = feat.std(1, keepdims=True) + 1e-6
    return ((feat - mu) / sd).astype(np.float32)


def firing_mask(counts96, n_active, upto=None):
    """Label-free 96-bool mask: 1 for the top-`n_active` electrodes by mean firing rate.

    `upto` restricts the ranking to the first `upto` bins (e.g. a calibration window); None = all."""
    fr = counts96[:, :upto].mean(1)
    sel = np.argsort(fr)[-n_active:]
    m = np.zeros(N_ELEC, dtype=np.float32)
    m[sel] = 1.0
    return m


def apply_mask_torch(neural192, mask96):
    """Zero the unselected electrodes in BOTH the raw and ewma blocks.

    neural192: (B, 192, T); mask96: (B, 96) or (96,). Returns (B, 192, T) with masked slots = 0."""
    import torch
    if mask96.dim() == 1:
        mask96 = mask96.unsqueeze(0).expand(neural192.shape[0], -1)
    m = torch.cat([mask96, mask96], dim=1).unsqueeze(-1)           # (B, 192, 1); raw+ewma share mask
    return neural192 * m


def make_masked_input_torch(neural192, mask96):
    """Full model input: [masked neural 192 ; mask 96 broadcast over T] -> (B, 288, T)."""
    import torch
    B, _, T = neural192.shape
    if mask96.dim() == 1:
        mask96 = mask96.unsqueeze(0).expand(B, -1)
    masked = apply_mask_torch(neural192, mask96)
    mask_ch = mask96.unsqueeze(-1).expand(B, N_ELEC, T)            # (B, 96, T) constant through window
    return torch.cat([masked, mask_ch], dim=1)                    # (B, 288, T)


IN_DIM = 2 * N_ELEC + N_ELEC                                       # 288

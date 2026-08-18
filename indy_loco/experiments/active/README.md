# Active Indy Loco experiment

## Phase 8: permitted neural-lookahead comparison

**Indy status: complete.** The 48 ms condition reached fold-macro test R²
`0.7576 ± 0.0396`; the 100 ms condition reached `0.7554 ± 0.0397`. The 48 ms
condition is slightly higher overall, while session-specific effects differ.

**Loco status: runner ready, not trained.**

`phase8_future_lookahead_fivefold.py` compares two deliberately non-causal
latency conditions on the three Indy sessions from the Phase 7 paper benchmark:

- nominal 50 ms, represented conservatively as 48 ms because the source has
  4 ms resolution;
- exact 100 ms.

Each condition performs five reach-level folds per session, for 30 independent
fits in total. Both conditions use identical fold assignments, fresh seed-43
weights, all 96 Indy channels, the frozen 64/64 TCN+GRU hyperparameters, and
0.20 paired channel dropout. Feature and target statistics are fit from
training reaches only. Validation selects each checkpoint; test inference is
performed only after selection.

The lookahead is implemented by aligning every 40 ms neural count bin later
than its corresponding 40 ms velocity target bin. The complete neural and
target intervals remain inside the same reach. Short reaches are right-padded
and masked, and reaches longer than eight seconds or unable to support the
largest lookahead are excluded identically from both conditions.

Validate the protocol:

```bash
python indy_loco/experiments/active/phase8_future_lookahead_fivefold.py \
  --validate-only
```

Run both conditions, all sessions, and all folds:

```bash
python indy_loco/experiments/active/phase8_future_lookahead_fivefold.py \
  --threads 4
```

Resume after interruption:

```bash
python indy_loco/experiments/active/phase8_future_lookahead_fivefold.py \
  --threads 4 --resume
```

Use `--lookahead future50ms` or `--lookahead future100ms` for one condition,
and `--session SESSION_NAME` for one session. CUDA is selected automatically
when available; otherwise the runner uses CPU. Apple MPS is disabled.

Outputs will be written under
`../../results/phase8_future_lookahead_fivefold/`. Phase 6 and Phase 7 remain
archived and their retained checkpoints are never overwritten.

## Loco extension

`phase8_loco_future_lookahead_fivefold.py` applies the same two lookahead
conditions to the three Loco paper sessions. The model still receives only 96
physical channels: each fold ranks the 192 Loco source channels using training
reaches only, then freezes the selected 96 for validation and test. The same
fold chooses the same channels for both lookahead conditions.

Validate:

```bash
python indy_loco/experiments/active/phase8_loco_future_lookahead_fivefold.py \
  --validate-only
```

Run all 30 Loco fits:

```bash
python indy_loco/experiments/active/phase8_loco_future_lookahead_fivefold.py \
  --threads 4
```

Resume:

```bash
python indy_loco/experiments/active/phase8_loco_future_lookahead_fivefold.py \
  --threads 4 --resume
```

Loco outputs are isolated under
`../../results/phase8_loco_future_lookahead_fivefold/` and cannot overwrite the
completed Indy Phase 8 evidence.

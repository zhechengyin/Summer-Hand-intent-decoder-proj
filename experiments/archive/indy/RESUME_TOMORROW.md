# Resume checklist — COMPLETE (2026-07-13)

All the experiments queued here (and their follow-ups iter15–18) have run. See
`../HANDOFF.md` for the current state and the one pending build step.

Outcome in one line: the deployable decoder is **strictly-causal** (bidirectional
isn't real-time), **TEST R² = 0.606**, ~73 KB int8, and it's near its ceiling —
every model-side lever was a dead end; only more (nearby) data helped.

**Next task:** `py models/tcn_gru_8ch/train_and_save.py` to write the causal
checkpoint (config is already `bidir=False`; the saved checkpoint is still the old
bidirectional one).

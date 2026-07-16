# Indy/Loco processing scripts

`build_bin_40ms_causal_counts.py` is the only supported pipeline. It creates
unsmoothed 40 ms spike counts and a causally filtered, backward-difference
velocity target.

Run scripts from the repository root. They read `data/raw/indy_loco/` and write
to `data/processed/indy_loco/bin_40ms_causal_counts/`.

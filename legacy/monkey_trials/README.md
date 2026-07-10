# Legacy intracortical trials

These scripts are the experiment history behind the current model in
[`../../models/`](../../models/). They are retained to reproduce ablations and
research decisions, but they are not the recommended model entry point.

| Script | Historical experiment |
| --- | --- |
| `velocity.py` | Within-session sorted-unit decoding |
| `nch.py` | 8/16/32/96-electrode cost sweep |
| `chan_select.py` | Random, firing-rate, and learned channel selection |
| `tune.py` | Rate smoothing and causal inference sweep |
| `vellp.py` | Velocity-target low-pass sweep |
| `subbin.py` | Within-bin spike timing versus counts |
| `slow_fast.py` | Slow/fast velocity decomposition |
| `activation.py` | Activation-function sweep |
| `multi.py` | Resumable per-session experiment runner |

Shared model code and the held-out cross-session evaluation were promoted to
`models/`; these trials import that package where needed.

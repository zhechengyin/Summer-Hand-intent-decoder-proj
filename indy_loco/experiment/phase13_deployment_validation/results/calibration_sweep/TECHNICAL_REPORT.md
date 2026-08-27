# Phase 13 Round 2: calibration-duration sweep

## Technical summary

Seven minutes is the recommended calibration duration for the current Midsize
deployment path. Exact CubeAI replay on the fixed common evaluation mask raised
the six-session macro mean R² from **0.5988 at 1 minute** to **0.6927 at 7
minutes** (`+0.0939`). Ten minutes reached **0.7060**, so the extra three minutes
after the knee bought only `+0.0134` mean R². All six sessions improved between
1 and 7 minutes.

The recommendation is a deployment tradeoff, not a claim that 7 minutes is the
mathematical optimum. It is the first duration after which every observed FP32
marginal gain was at most `0.005 R²/min`. A second sweep whose longest duration
was 8 minutes independently selected 7 minutes and retained at least 648 common
test bins per session.

## Primary finding

The fixed-mask FP32 curve rises from `0.5988` at 1 minute to `0.6938` at 7
minutes and `0.7062` at 10 minutes. Seven minutes captures **88.4%** of the
observed 1-to-10-minute gain; 9 minutes is required to capture at least 95%.
The marginal changes after 7 minutes are `+0.0047`, `+0.0044`, and `+0.0033`
R² per added minute, all below the predeclared `0.005` plateau threshold.

| Calibration | FP32 common-mask R² | Exact CubeAI R² | Delta vs 1 min |
|---:|---:|---:|---:|
| 1 min | 0.5988 | 0.5988 | — |
| 7 min | 0.6938 | 0.6927 | +0.0939 CubeAI |
| 9 min | 0.7029 | — | +0.1041 FP32 |
| 10 min | 0.7062 | 0.7060 | +0.1073 CubeAI |

## Six-session CubeAI confirmation

| Session | Common bins | 1 min R² | 7 min R² | 10 min R² | 7−1 min | 10−7 min |
|---|---:|---:|---:|---:|---:|---:|
| indy_20160622_01 | 3314 | 0.7329 | 0.7791 | 0.7904 | +0.0461 | +0.0113 |
| indy_20160630_01 | 1820 | 0.5153 | 0.6179 | 0.6236 | +0.1026 | +0.0057 |
| indy_20170131_02 | 353 | 0.5365 | 0.6547 | 0.6788 | +0.1182 | +0.0241 |
| loco_20170210_03 | 1688 | 0.6839 | 0.7217 | 0.7411 | +0.0377 | +0.0195 |
| loco_20170215_02 | 1143 | 0.3665 | 0.6053 | 0.6229 | +0.2388 | +0.0176 |
| loco_20170301_05 | 456 | 0.7574 | 0.7773 | 0.7793 | +0.0199 | +0.0020 |

The paired 1-to-7-minute CubeAI differences are positive for all six sessions:
mean `+0.0939`, median `+0.0744`, range `+0.0199` to `+0.2388`. An exact
two-sided Wilcoxon signed-rank test gives `p=0.03125`; with only six sessions,
this is supporting evidence rather than a population-level guarantee. The
largest contribution comes from `loco_20170215_02`, but the direction remains
positive after removing that session.

## Scope, data, and metric definition

- Model path: current Midsize deployment checkpoint selected for each session,
  using the firmware-equivalent continuous 50-bin window and causal calibration.
- Sessions: the six canonical Indy/Loco GUI sessions.
- Calibration data: only the unlabeled chronological prefix of each session;
  no future input, target, label, residual, or memory-bank value is used.
- Primary metric: session-macro mean of output-averaged R² on the same best-fold
  test bins for every duration. In the primary sweep the bins must occur after
  10 minutes, the longest tested calibration.
- Secondary metric: R² on every test bin available after each duration. It is
  retained for diagnosis but is not used to select the duration because the
  evaluated population changes with duration.

## Methodology

The sweep tested `0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10` minutes. For every
duration and session, the script recomputed the firmware-style calibration mean
and standard deviation from the permitted prefix, normalized the full stream,
then evaluated the selected best-fold test checkpoint. FP32 was used for the
complete sweep. The 1-, 7-, and 10-minute points were independently replayed
through the exported CubeAI numeric path.

The knee rule selects the earliest duration after which all later adjacent
slopes are no greater than `0.005 R²` per additional minute. A separate rule
reports the earliest duration reaching 95% of the best observed gain over the
1-minute baseline.

## Robustness and limitations

The 10-minute common mask is deliberately strict but leaves only 353 test bins
for the shortest session and 456 for another. The sensitivity sweep capped the
maximum at 8 minutes, leaving at least 648 common bins per session. It still
selected 7 minutes: R² was `0.6105` at 1 minute, `0.7020` at 7 minutes, and
`0.7063` at 8 minutes; 7 minutes captured **95.5%** of the observed 1-to-8-minute
gain.

This is a retrospective six-session replay. A live user may move differently
during calibration, and prefix duration is confounded with the amount and
variety of movement observed. The current result therefore supports a firmware
A/B candidate, not a universal guarantee. The firmware remains hard-coded to
1 minute until that change is implemented and tested on-device.

## Recommended next steps

1. Add a configurable calibration duration to firmware/GUI and set the test
   default to 7 minutes; do not silently change the production default yet.
2. Verify 1 versus 7 minutes on-board for all six packaged sessions, including
   calibration statistics, CubeAI outputs, latency, SRAM/flash, and watchdog
   behavior.
3. For a more usable product, evaluate an adaptive stop rule after 3 minutes:
   stop when channel mean/std and prediction quality proxies stabilize, with a
   hard cap at 7 minutes.
4. Run the same sweep on genuinely unseen sessions/users before treating 7
   minutes as a general deployment setting.

## Further questions

- Does movement diversity matter more than elapsed time? A guided calibration
  protocol may reach the 7-minute quality sooner.
- Can the calibration estimator use robust or exponentially weighted statistics
  to reduce sensitivity to the first minute?
- Does the same knee hold for Large and for GRU external-memory correction, or
  should normalization calibration and memory-bank collection be timed
  separately?

## Reproducibility

Primary outputs are `calibration_sweep.json`,
`calibration_sweep_summary.csv`, and `calibration_sweep_by_session.csv` in this
directory. The 8-minute sensitivity outputs are under `sensitivity_max8/`. The
runner is `../../run_calibration_sweep.py` relative to this report.

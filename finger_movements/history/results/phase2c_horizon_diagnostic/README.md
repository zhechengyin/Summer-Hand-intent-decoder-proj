# Phase 2c Causal Horizon Diagnostic

This corrected-data experiment measured how classification changed as causal history accumulated from a known epoch start.

| Available history | Mean OOF BA |
|---:|---:|
| 50 ms | 50.62% |
| 100 ms | 51.27% |
| 150 ms | 55.16% |
| 200 ms | 56.33% |
| 250 ms | 62.97% |
| 300 ms | 68.97% |
| 350 ms | 72.89% |
| 400 ms | 77.22% |
| 450 ms | 79.65% |
| 500 ms | 82.93% |

Temporal filters ran left-to-right, and perturbing samples after the prediction point changed the current output by exactly zero. The later streaming experiment corrected the operational interpretation: the window should be understood as past context ending now, not as a delay after a fixed point A.

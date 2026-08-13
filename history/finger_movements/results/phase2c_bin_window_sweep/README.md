# Phase 2c Bin and Window Sweep

Corrected official MATLAB TRAIN data; seeds 42–44; five folds per seed; official TEST not loaded.

| Past window | Mean OOF BA |
|---:|---:|
| 200 ms | 79.62% |
| 300 ms | 79.43% |
| 400 ms | 83.99% |
| 500 ms | 82.93% |

The 400 ms window was selected because it improved mean BA, worst-seed BA, and seed stability over 500 ms. Bins of 10, 20, 50, and 100 ms produced equivalent endpoint features; 50 ms remained the deployment update interval.

# Phase 4b Five-Seed Confirmation

This experiment compared the 64/64 reference and 48/48 candidate over seeds 42–46 and five leave-one-month-out folds (50 fits).

| Architecture | Selection score | Macro R² | 10th-percentile R² | Worst R² | Parameters |
|---|---:|---:|---:|---:|---:|
| 64/64 | 0.4773 | 0.5500 | 0.2592 | -0.1600 | 78,786 |
| 48/48 | 0.4738 | 0.5441 | 0.2629 | -0.1746 | 45,266 |

The 48/48 model passed all predefined non-inferiority limits and reduced parameter and multiply counts by about 42%. It was promoted as the preferred standalone firmware candidate. Checkpoint construction is recorded separately in the main experiment log.

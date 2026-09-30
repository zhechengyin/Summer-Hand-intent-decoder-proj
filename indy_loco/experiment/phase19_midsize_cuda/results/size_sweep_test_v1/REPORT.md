# Frozen width-sweep test evaluation / 宽度扫描 test 结果

| Width | Test R² mean ± sample SD | Paired Δ vs Midsize | Wins / 30 |
|---|---:|---:|---:|
| 64 | 0.741138 ± 0.065598 | +0.000000 | 0 |
| 96 | 0.739618 ± 0.065505 | -0.001520 | 12 |
| 128 | 0.743070 ± 0.066986 | +0.001932 | 17 |
| 256 | 0.741075 ± 0.062379 | -0.000062 | 13 |

Canonical warm-started Midsize re-evaluated on the same evaluation backend. Wider models received additional validation-selected optimization. This does not isolate width from extra training. Test scores did not choose checkpoints or change the frozen validation ranking.

基线和扩宽模型均有 warm-start；扩宽模型还经过额外训练，因此不能把差值完全归因于宽度。
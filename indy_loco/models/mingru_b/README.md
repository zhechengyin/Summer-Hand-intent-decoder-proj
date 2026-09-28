# Phase18 retained model B / 保留的 B 模型

Validation selected this model before final test evaluation. All 30 selected
EMA checkpoints are retained: six sessions, five folds, seed 43. No checkpoint
was selected by its test score. These are fold-specific models, not an ensemble
or a new model fitted to all available data.

本模型在 test 评估前由 validation 选出。保留六个 session、五折、seed 43
的全部 30 个 EMA 检查点；没有按 test 分数挑选单折。这是分折模型集合，
不是集成模型，也不是另外训练的全数据模型。

| Metric / 指标 | Result / 结果 |
|---|---:|
| Validation R², 30-fold mean | 0.7614946 |
| Test R², mean ± sample SD | 0.7567263 ± 0.0689514 |
| Paired test change vs retained r07 | +0.0158063; 25/30 wins |
| Paired test change vs historical Midsize | +0.0155887; 22/30 wins |
| Parameters | 374,402 |
| FP32 weight bytes, excluding metadata/activations | 1,497,608 |
| Dense MACs per 50-bin window | 13,647,616 |

## Model and input contract / 模型与输入

Width 128, three minGRU blocks with expansion 2, residual causal depthwise-k5
and pointwise frontend, and a 128→256→2 output head. Only the final output is
returned. Earlier recurrent history remains necessary. Independent windows
reset state; there is no persistent streaming. Parallel and sequential
recurrence paths are included. MCU kernels, SDRAM placement and measured
board latency are not part of this package.

Input is **already preprocessed** float32 `[batch, 192, 50]`: 96 selected physical
channels, counts plus EWMA features, 40-ms bins and the frozen seven-minute
calibration contract. Preserve the exact feature order and preprocessing from
Phase16/17; do not feed raw spikes directly or fit new scalers to test data.
The fold checkpoint includes selected channels, calibration mean/effective
standard deviation and target scalers. `manifest.json` retains preprocessing
and split fingerprints; it does not contain the source dataset.

输入必须是已经按原协议处理的 float32 `[batch, 192, 50]`：96 个选定通道，
counts 与 EWMA 特征，40-ms bin 和固定七分钟校准。保持原特征顺序及预处理，
不能直接输入原始 spike，也不能在 test 上重新拟合 scaler。检查点含通道选择、
校准及目标 scaler；manifest 保存预处理和数据划分指纹，不包含原始数据。

```python
import torch
from indy_loco.models.mingru_b.load import load_fold, predict_velocity

model, scalers = load_fold("indy_20160622_01", 1)
# Replace this example with windows produced by the frozen preprocessing.
normalized_windows = torch.zeros(1, 192, 50)
velocity = predict_velocity(model, normalized_windows, scalers)  # [1, 2]
```

The loader verifies SHA-256 and uses `torch.load(weights_only=True)`. Checkpoints
contain only the selected EMA weights and inference metadata, not raw/alternative
weights, optimizer states or training datasets. Model code imports no local
experiment or archived implementation.

## Decision, status and experiment log / 决策、状态与实验记录

- 2026-09-28: Phase18 completed 72 fits and 60 A/B test-fold evaluations.
  B was selected on validation; its globally frozen recipe is LR 0.001 + EMA.
- 2026-09-28: User requested publishing only the latest B. Exported all 30
  selected EMA states with strict state loading and exact prediction equality
  against the original implementation on fixed test inputs. Source checkpoint
  hashes and the original configuration/selection/metrics hashes are retained.
- The earlier r07 baseline and its backup remain local and unchanged. Other
  candidate models, datasets and backup archives are not part of this release.
- No next numbered phase, additional training or deployment is started here.

2026-09-28 完成 Phase18，并按用户要求只发布 B。旧 r07 及备份仍在本地保留；
此次不发布其他候选模型或数据，也不启动下一编号阶段、额外训练或板端部署。

## Verification and limits / 核验与限制

```powershell
.\.venv\Scripts\python.exe -m unittest indy_loco.models.mingru_b.test_package -v
```

The check covers hashes, strict loading, parameter count, independent-window
reset and sequential/full-reference equivalence for every checkpoint. See
`verification.json` for the export comparison, `training_recipe.json` for the
recipe and `manifest.json` for fold-level metrics and provenance.

One seed does not establish seed robustness. The six sessions and overlapping
folds are not 30 independent recordings or nested cross-validation. Historical
test results had already been exposed; historical Midsize was warm-started,
whereas this model trained from scratch. Published R² describes this benchmark,
not measured STM32 inference quality or latency.

单 seed 不构成种子稳定性验证；六个 session 的重叠折不是 30 次独立记录或
嵌套交叉验证。历史 test 已暴露，Midsize 使用 warm start，本模型从头训练。
这些 R² 是当前基准结果，不代表已经测量过的 STM32 推理精度或延迟。

# Phase19: Midsize equivalent CUDA experiments / 等价 CUDA 实验

Completed on 2026-09-30. CUDA verification accepted epilogue, causal and
LayerNorm/ReLU candidates; GRU and combined failed the AdamW-update gate and
were not timed. LayerNorm/ReLU reduced mean full-epoch time by 9.28%, below the
prespecified 10% adoption threshold. The original implementation remains default.
See [CUDA report](results/run_20260929_221150/REPORT.md).

The [paired width sweep](size_sweep/README.md) completed all 90 training jobs.
Frozen test evaluation, including re-evaluation of original Midsize, completed
120 checkpoints on CPU: test R2 0.741138 (64), 0.739618 (96), 0.743070 (128),
0.741075 (256), each averaged over the same 30 folds. These are warm-started
comparisons with additional training for wider models, not width-only effects.
See [test report](results/size_sweep_test_v1/REPORT.md). All 90 sweep checkpoints
and test predictions are retained; original checkpoints were not changed.

已完成 CUDA 对照、90项宽度训练及120项 test 检查点评估。LayerNorm/ReLU
完整 epoch 节省9.28%，未达到预设10%标准；原实现保持默认。128/128 的
test R2 小幅提高，96/96 小幅下降，256/256 基本持平。详见上述结果报告。

## Run / 运行

In PowerShell, from any directory / 在 PowerShell 中执行：

```powershell
Set-Location "C:\Users\fangz\Documents\Summer-Hand-intent-decoder-proj"
.\.venv\Scripts\python.exe -X utf8 -u -m indy_loco.experiment.phase19_midsize_cuda.benchmark --epochs 3 --repeats 3
```

This first compiles with torch-bundled NVRTC, then checks each candidate on CUDA.
Only passing candidates enter timing. No nvcc/MSVC installation is requested.
Uses the existing Windows NVRTC loader pattern from Phase18, with an independent
source file and runtime. CUDA is required; there is no CPU fallback.

先用 PyTorch 自带 NVRTC 编译，再检查 CUDA 数值，仅通过的候选进入计时。
不需要另装 nvcc/MSVC。仅支持 Windows、CUDA、FP32、eager、一阶 autograd。

Optional checks only / 可选：只做正确性检查、不训练：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m indy_loco.experiment.phase19_midsize_cuda.benchmark --verify-only
```

To test a subset / 只测试部分候选：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m indy_loco.experiment.phase19_midsize_cuda.benchmark --variants causal layernorm --epochs 3 --repeats 3
```

## Arms / 对照组

| Name | Replacement / 替换内容 |
|---|---|
| pytorch | Original Phase16 BASELINE Midsize training model, native nn.GRU; 原生 PyTorch/cuDNN 基线 |
| epilogue | Original padded convolution + custom crop/residual/ReLU forward/backward; 只融合卷积后处理 |
| causal | Left-pad + unpadded vendor convolution + custom residual/ReLU; 左补零避免丢弃右侧卷积输出 |
| layernorm | Custom 64-channel LayerNorm/ReLU forward/input backward, native parameter-gradient reductions; 融合归一化激活 |
| gru | Packed input projection, per-step packed recurrent projection, custom gates/state-update forward/backward; 三个分支全部保留 |
| combined | causal + layernorm + gru; 联合候选 |

Convolution/GEMM arithmetic and their gradients stay in PyTorch/cuDNN/cuBLAS.
The causal arm is **not** a handwritten convolution microkernel: only its padding
and output extent change, with a custom fused epilogue. Explicit padding cost is
included. The GRU arm is an experimental Python time loop with native GEMMs and
custom pointwise CUDA. cuDNN already packs gates and may be faster. No speedup is
assumed. The combined arm is fixed in advance, not selected using timing results.

卷积和矩阵乘法保留成熟后端。因果卷积组不是手写卷积主循环；它改变补零方式和
输出范围，再融合后处理，计时包含显式补零成本。GRU 组仍有 Python 时间循环和
多次 GEMM，可能比 cuDNN 慢。联合组是预先固定的组合，不保证加速。

All arms retain 86,978 parameters, kernel3/dilations1,2,4,8, GRU hidden64,
LayerNorm eps1e-5, paired channel dropout0.20, pre-GRU dropout0.10, all 50 hidden
states and all output-head positions. Reset-after-projection GRU semantics,
including recurrent candidate bias, are preserved. No persistent streaming,
quantization, TF32, AMP, gate removal, activation approximation or head pruning.
Floating-point reduction order can differ; mathematical equivalence alone does
not establish numerical equivalence.

所有组均保留 86,978 参数、窗口50、kernel3、dilation1/2/4/8、GRU hidden64、
原 dropout、完整序列输出。无持久状态、混合精度、TF32 或删门。浮点归约顺序
可能不同，因此必须先通过数值检查。

## Data and timing / 数据与计时

- Single seed43, session `indy_20160622_01`, fold1. Reuse canonical fold1 weights
  identically in every arm. This is a warm-started timing experiment, not a
  reproduction of historical from-start training or a new final model.
- Reuse frozen Phase13/16 preprocessing through Phase17's verification helpers.
  Validate protocol locks, source data identity, fold membership, channel choice,
  and canonical scalers. Never evaluate test predictions/metrics. Dataset loading
  and metadata verification still include the original session arrays.
- AdamW: recurrent/head LR3e-4, encoder LR7.5e-5, weight decay0.025, clip1.0,
  batch128 including the short tail. Cosine T_max remains20 for a 3-epoch prefix.
  Early stopping is disabled for equal timing budgets; no EMA is added.
- Three warm-up forward/backward batches; restore weights/RNG before training.
  Compilation, data loading and warm-up are outside epoch timing. Time training,
  validation, and full epochs separately with CUDA synchronization. Include the
  original CPU window construction/transfers, scalar synchronization, validation
  and best-state CPU copies; no GPU dataset caching advantage in either arm.
- Three timing repeats of the same seed, rotated/reversed arm order. By default,
  up to six arms x three epochs x three repeats = 54 epochs. Failed candidates
  are skipped. Repeats are not independent accuracy seeds, and this ordering does
  not eliminate thermal drift on a laptop.
- Shares the Phase17/18 cooperative GPU lock. Fails if another cooperating run
  owns it. Keep other GPU workloads closed for interpretable timings.
- Output uses a unique directory per invocation. No automatic resume, overwrite,
  model promotion, commit, push, recurring monitor or old training restart.

单 seed43、第一 session 第1折，所有组从同一个已保存 Midsize 开始。复用冻结
预处理并验证，不计算 test。默认每组3 epoch、重复计时3次，最多54 epoch。
重复不代表独立精度 seed。完整计时包含 CPU 构造窗口、传输、验证及日志。
各次运行独立保存；若中断，保留证据，不把中断前后计时自动混合。

## Gates and outputs / 检查与结果

`verification.json` records operator outputs/gradients, model train/eval outputs,
all parameter/input gradients and one clipped AdamW update on a real batch.
Shapes cover batch1, batch7, batch128 and sequence lengths1,17,50; operator checks
include zero/saturated inputs and exact ReLU zero. Tolerances are fixed in
`verify.py`: output atol2e-5/rtol2e-4, gradient atol3e-5/rtol3e-4, update
atol2e-6/rtol2e-5. A failure records its exact stage and skips that arm's training;
CUDA/runtime failures abort rather than continuing with a potentially bad context.

`status.json` shows the current stage. `repeat*_*.epochs.json` holds live epoch
logs; the final `repeat*_*.json` files include full timing. `summary.json` and
`REPORT.md` report mean epoch time, sample SD across repeat means, peak VRAM,
paired speedups and maximum matched-epoch validation R² deviation. All remain
inside `phase19_midsize_cuda/results/run_*`.

Tentative adoption requires >=3 epochs/repeats, >=10% mean full-epoch time savings,
every paired repeat faster, and <=0.001 validation R² deviation at every matched
epoch. Even passing is only a short-run candidate; no automatic replacement and
no claim about final test R² or STM32 inference speed.

输出含数值检查、实时 epoch 日志、平均耗时、重复均值的样本标准差、显存峰值、
逐次加速比及 validation R² 偏差。平均完整 epoch 至少节省10%、每次都更快、
所有匹配 epoch 的 validation R² 差不超过0.001，才列为短程候选。
结果不能证明最终 test R² 等价，也不能直接代表 STM32 部署收益。

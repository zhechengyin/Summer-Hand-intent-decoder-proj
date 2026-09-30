# minGRU A/B comparison / minGRU A/B 对照

This experiment compares two compact-head minGRUs after the diagnostic analysis
has completed. Both retain width 128, three minGRU blocks, FFN expansion 2, and
a 128 → 256 → 2 output head. **A** uses the existing temporal core. **B** adds
a residual causal convolution after the stem dropout: layer normalization,
depthwise convolution with kernel 5, GELU, and a pointwise width-128 projection.
Only past and current bins enter the convolution. This comparison changes the
head and, for B, adds local temporal processing; it is separate from the prior
large-head model and preserves its code and results.

本实验在诊断分析完成后，对两个较小输出头的 minGRU 进行对照。两者均保留
宽度 128、三个 minGRU block、FFN 扩展倍数 2，输出头改为 128 → 256 → 2。
**A** 使用原时序核心；**B** 在 stem dropout 后加入残差因果卷积，依次为
LayerNorm、核大小 5 的逐通道卷积、GELU 和宽度 128 的逐点投影。
卷积仅使用当前及过去的 bin。本轮缩小输出头，并在 B 中增加局部时序处理，
作为独立实验保留此前大输出头模型的代码与结果。

| Model / 模型 | Parameters / 参数量 | FP32 weights / 权重 | Dense MACs per window / 每窗口密集 MAC |
|---|---:|---:|---:|
| A | 356,866 | 1.427464 MB | 12,796,416 |
| B | 374,402 | 1.497608 MB | 13,647,616 |

Weights exclude activations and workspace. MACs omit normalization, nonlinearities,
elementwise recurrence, and memory movement; they are not measured board latency.

权重空间不包括激活和工作区。MAC 不包括归一化、非线性函数、逐元素递推及
内存搬运，不能作为板端实际延迟。

The required diagnostic receipt is
[`ab_diagnostics_v1/summary.json`](../results/ab_diagnostics_v1/summary.json).
Its existence is checked by the launcher, and the training runner verifies the
diagnostic evidence before fitting. A diagnostic process and a training process
must not use the GPU concurrently.

启动前需要诊断记录
[`ab_diagnostics_v1/summary.json`](../results/ab_diagnostics_v1/summary.json)。
启动脚本检查该文件是否存在，训练程序在拟合前核验诊断证据。诊断进程与训练
进程不得同时使用 GPU。

## Fixed training and EMA / 固定训练与 EMA

Both architectures use two learning rates, **0.001 and 0.0006**, with the same
rate in all parameter groups. All fits use seed 43, FP32, AdamW, batch size 128,
gradient clipping 1.0, a 60-epoch cosine schedule, and at most 60 epochs. Weight
decay is 0.01 outside the output head and 0.03 in the head. Channel and post-stem
dropout remain 0.1; head dropout is zero. The existing six sessions, fivefold
split, preprocessing, 40-ms bins, and independent zero-initialized 50-bin windows
are unchanged. There is no persistent streaming.

两种架构均测试 **0.001 和 0.0006** 两个学习率，各参数组使用相同学习率。
所有训练均使用 seed 43、FP32、AdamW、batch size 128、梯度裁剪 1.0、
周期为 60 epoch 的余弦衰减，最多训练 60 epoch。输出头 weight decay 为
0.03，其他参数为 0.01；通道与 stem 后 dropout 均为 0.1，输出头 dropout
为零。原六个 session、五折划分、预处理、40-ms bin 和独立零初始状态的
50-bin 窗口保持不变，不采用跨窗口持续状态。

Each fit maintains two weight policies from one optimizer trajectory: **raw**
weights and **EMA** weights. EMA starts as a copy of the initial raw weights and
updates after each optimizer step with decay 0.99. It does not update the raw
model or create another training fit. Raw and EMA each retain their own best
checkpoint by validation normalized MSE. Stop after 15 consecutive epochs in
which neither policy improves its own best validation loss, or at epoch 60.
EMA adds evaluation and storage overhead; it does not double the fit budget.

每次训练在同一条优化轨迹上维护两种权重策略：**raw 原始权重**和 **EMA
指数滑动平均权重**。EMA 从初始原始权重的副本开始，每次 optimizer step
后以 0.99 的衰减系数更新，不修改原始模型，也不额外训练一遍。raw 和 EMA
分别按验证集标准化 MSE 保存各自最佳检查点。连续 15 个 epoch 两种策略均未
改善各自最佳验证损失时停止，或最迟在第 60 个 epoch 停止。EMA 会增加评估
和存储开销，但不使训练次数翻倍。

## Selection and complete A/B evaluation / 选择与完整 A/B 评估

Screen both learning rates for both architectures on fold 1 of each session:
**2 architectures × 2 learning rates × 6 sessions = 24 fits**. Using only these
validation results, select and freeze one **learning-rate/weight-policy pair
for each architecture**. The policy is global for that architecture, not chosen
separately for each fold.

先在每个 session 的第 1 折测试两种架构、两个学习率，共
**2 种架构 × 2 个学习率 × 6 个 session = 24 次训练**。只使用这些验证结果，
为每种架构分别选择并冻结一组 **学习率/权重策略**。权重策略对该架构全局统一，
不能逐折选择表现较好的 raw 或 EMA。

Complete folds 2–5 for each architecture's frozen learning rate:
**2 × 6 × 4 = 48 additional fits, 72 total**. Retain the screening-selected raw
or EMA policy throughout confirmation; do not reselect it using the remaining
folds. Compare the architectures using their complete 30-fold validation results.
After all 72 fits and both configurations are frozen, evaluate **both A and B
on their 30 test folds: 60 test evaluations**. The validation-selected architecture
remains the selected architecture regardless of the subsequent test ranking.

再按各架构已冻结的学习率完成第 2～5 折，新增 **2 × 6 × 4 = 48 次，
总计 72 次训练**。确认阶段始终使用筛选阶段选定的 raw 或 EMA，不根据后续
折重新选择策略。用两种架构各自完整的 30 折验证结果比较架构优劣。
全部 72 次训练完成并冻结配置后，**A、B 各评估 30 个测试折，共 60 次测试
评估**。最终架构选择保持由验证集决定，不因后续测试排名而改变。

The test set never chooses learning rate, checkpoint, weight policy, or
architecture. Reporting both architectures is the requested A/B comparison,
not a test-based selection rule. Training-set diagnostic losses, when recorded,
also do not participate in checkpoint or configuration selection.

测试集不用于选择学习率、检查点、权重策略或架构。汇报两种架构是为了完成
A/B 对照，不是按测试集选择优胜者。即使记录训练集诊断损失，也不将其用于
检查点或配置选择。

## Run and stop / 运行与停止

After the diagnostic analysis completes, launch or resume from PowerShell:

诊断完成后，在 PowerShell 中启动或恢复：

```powershell
Set-Location 'C:\Users\fangz\Documents\Summer-Hand-intent-decoder-proj'
& '.\indy_loco\experiment\phase18_large_mingru\ab_study\start_comparison.ps1'
```

The launcher checks for active Phase17/Phase18 Python processes and refuses to
bypass this run's stop marker. It uses the repository virtual environment,
explicit working directory, hidden background process, and unique stdout/stderr
logs. It adds `--resume` only when this run has a saved configuration. The runner
holds operating-system locks and verifies fingerprints before resuming completed
fits. An interrupted incomplete fit restarts. Do not edit fingerprinted Python
files after launching.

启动脚本会检查是否有 Phase17/Phase18 Python 进程运行，不会绕过本轮停止
标记。它使用仓库虚拟环境、明确工作目录、隐藏后台进程和唯一 stdout/stderr
日志；仅在本轮已有配置文件时添加 `--resume`。训练程序持有操作系统锁并核验
指纹后复用已完成训练；被中断且未完成的一次训练会重新开始。启动后不要修改
纳入指纹校验的 Python 文件。

Results are under `indy_loco/experiment/phase18_large_mingru/results/ab_comparison_v1`.
Inspect `progress.json`, per-fit epoch logs, checkpoints, and the final report.
Successful completion requires complete metrics and 60 verified A/B test-fold
results, not merely process exit. To stop after the current fit saves:

结果位于 `indy_loco/experiment/phase18_large_mingru/results/ab_comparison_v1`。
可查看 `progress.json`、每次训练的 epoch 日志、检查点和最终报告。成功完成
要求完整指标及 60 个经过核验的 A/B 测试折结果，不能仅根据进程退出判断。
若需在当前一次训练保存后停止：

```powershell
New-Item -ItemType File -Force -Path '.\indy_loco\experiment\phase18_large_mingru\results\ab_comparison_v1\STOP_AFTER_FOLD'
```

Remove only this marker when intentionally resuming this comparison; leave
earlier experiments' stop markers and results unchanged.

仅在决定恢复本轮对照时删除本轮标记，不改动之前实验的停止标记或结果。

## Interpretation / 结果解释

Report each architecture's test R² mean and sample standard deviation, paired
A/B differences, its frozen learning rate and weight policy, parameter count,
and FP32 weight size. A single seed is not a seed-robustness study. Historical
tests are already exposed; globally tuned overlapping folds are not nested
cross-validation, and six sessions are not 30 independent recordings. Historical
Midsize was warm-started, while these models train from scratch. GPU runtime and
dense MAC counts do not establish STM32 latency or SDRAM transfer performance.

分别汇报两种架构的测试 R² 均值与样本标准差、A/B 配对差值、冻结的学习率与
权重策略、参数量和 FP32 权重大小。单一 seed 不构成随机种子稳定性实验。
历史测试结果已经公开，具有重叠的折上进行全局调参不等于嵌套交叉验证，六个
session 也不等于 30 次独立记录。历史 Midsize 使用过 warm start，而本轮
模型从头训练。GPU 耗时和密集 MAC 数量不能证明 STM32 延迟或 SDRAM 搬运性能。

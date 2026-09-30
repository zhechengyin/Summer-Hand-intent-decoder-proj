# Large minGRU regularization tuning / 大容量 minGRU 正则化调参

This follow-up tests whether regularizing the large output head improves
validation R². The width-128, three-block minGRU and 128 → 768 → 1024 → 2 head
retain **1,211,906 parameters** and the same inference computation. Head dropout
is active only during training. The experiment does not extend the training
budget or introduce persistent streaming.

本轮实验检验针对大输出头的正则化能否提高验证集 R²。模型仍为宽度 128、三个
minGRU block，输出头为 128 → 768 → 1024 → 2，共 **1,211,906 个参数**，
推理计算结构不变。输出头 dropout 仅在训练时生效；不延长训练上限，也不采用
跨窗口持续状态。

The original winning configuration reached its best validation checkpoint at
median epoch **17.5/60** across 30 folds. Mean validation normalized MSE worsened
from **0.23825** at the selected checkpoints to **0.25477** at the final epochs,
while training optimization loss fell from **0.06403** to **0.02922**. All 30
folds showed this direction of change, and 28 stopped before epoch 60. These
curves support studying regularization rather than simply training longer.
Training loss uses dropout, so its absolute difference from validation loss is
not a clean estimate of the generalization gap.
See the [learning-curve analysis](../results/regularization_v2_analysis/REPORT.md).

原优胜配置在 30 折中的最佳验证检查点对应 epoch 中位数为 **17.5/60**。
平均验证集标准化 MSE 从最佳检查点的 **0.23825** 上升到末轮的 **0.25477**，
同期训练优化损失从 **0.06403** 降到 **0.02922**；30 折均出现这一变化方向，
其中 28 折提前停止。因此，更值得测试正则化，而非单纯延长训练。训练损失包含
dropout 的影响，不能直接把它与验证损失的差值当作严格的泛化误差估计。
详见[学习曲线分析](../results/regularization_v2_analysis/REPORT.md)。

The output head contains **888,578 parameters (73.32%)** and previously had no
dropout of its own. This makes it a reasonable regularization target, but the
curves do not prove that the head alone caused overfitting. The original stronger
global-regularization finalist did not beat the anchor on all 30 validation
folds; adding regularization is not guaranteed to help.

输出头包含 **888,578 个参数，占 73.32%**，此前没有专门的 dropout，因此适合
作为局部正则化对象。但学习曲线不能证明过拟合仅由输出头造成。上一轮整体正则化
更强的候选在完整 30 折验证中未超过基准；加强正则化不保证提高 R²。

## Fixed design / 固定方案

All trials train from scratch with seed 43, FP32, AdamW, batch size 128, gradient
clipping 1.0, a 60-epoch cosine learning-rate schedule, at most 60 epochs, and
patience 15. Channel dropout and post-stem dropout remain 0.1; stem learning-rate
scale is 1.0. Head weight decay applies to the actual AdamW head parameter group;
the remaining parameters use weight decay 0.01. Head dropout is applied after
each of the two hidden GELU activations.

所有配置从头训练，使用 seed 43、FP32、AdamW、batch size 128、梯度裁剪 1.0、
周期为 60 epoch 的余弦学习率衰减、最多 60 epoch 和 patience 15。
通道 dropout 与 stem 后 dropout 均保持 0.1，stem 学习率倍率为 1.0。
输出头 weight decay 实际作用于 AdamW 的输出头参数组，其余参数使用 0.01。
输出头两处隐藏层 GELU 后各加入指定概率的 dropout。

Each epoch also records evaluation-mode loss on a fixed sample of at most
1,024 training bins, with dropout disabled. This provides a clearer diagnostic
alongside the training optimization loss; neither diagnostic selects a
checkpoint or configuration. The sample never includes validation or test bins.

每个 epoch 还会在固定抽样的最多 1,024 个训练 bin 上关闭 dropout，记录评估
模式的训练损失，辅助区分随机正则化影响与泛化差距。该诊断和训练优化损失均不
参与检查点或配置选择，抽样不包含验证集或测试集 bin。

| Trial / 配置 | Learning rate / 学习率 | Non-head weight decay / 其他参数 | Head weight decay / 输出头 | Head dropout / 输出头 dropout |
|---|---:|---:|---:|---:|
| r00: rerun anchor / 重新训练基准 | 0.001 | 0.01 | 0.01 | 0.0 |
| r01 | 0.001 | 0.01 | 0.01 | 0.1 |
| r02 | 0.001 | 0.01 | 0.01 | 0.2 |
| r03 | 0.001 | 0.01 | 0.01 | 0.3 |
| r04 | 0.001 | 0.01 | 0.03 | 0.1 |
| r05 | 0.001 | 0.01 | 0.1 | 0.1 |
| r06 | 0.0006 | 0.01 | 0.01 | 0.2 |
| r07 | 0.001 | 0.01 | 0.03 | 0.0 |

The anchor is rerun under this experiment's own provenance; no earlier fitted
checkpoint is imported or relabeled. Screen all eight configurations on fold 1
of each of six sessions: **48 fits**. Complete folds 2–5 for the top two:
**48 additional fits, 96 total**. Select checkpoints by minimum validation
normalized MSE and rank configurations by macro validation R². Freeze the final
configuration after its 30-fold validation evaluation; only then evaluate its
30 test folds. Test results never select configurations. No extra seeds are run.

基准配置在本实验中重新训练，不导入或改名复用之前已训练的检查点。
先在六个 session 各自的第 1 折筛选八组配置，共 **48 次训练**；再对前两名
完成第 2～5 折，新增 **48 次，总计 96 次训练**。每次训练按最低验证集
标准化 MSE 选择检查点，按平均验证集 R² 排名配置。完成 30 折验证后冻结优胜
配置，再评估它的 30 个测试折。测试结果不参与配置选择，本轮不增加其他 seed。

The original six sessions, five folds, channel selection, calibration, scaling,
40-ms bins, and 50-bin independent zero-state windows remain frozen and verified.
Earlier Phase17/Phase18 code and results remain unchanged. Head dropout vanishes
in evaluation, so the existing sequential-arithmetic equivalence checks and
discarded-computation removal still apply. Software batched weight reuse is
preserved; physical SDRAM transfers and board latency are not tested here.

原六个 session、五折划分、通道选择、校准、标准化、40-ms bin 及 50-bin
独立零初始状态窗口均保持固定并接受核验。此前 Phase17/Phase18 的代码和结果
保持原状。评估时输出头 dropout 不生效，因此原顺序算术等价性检查与无效计算
删除仍适用。继续保留软件层面的批量权重复用；本实验不测试物理 SDRAM 搬运
或板端延迟。

## Run and stop / 运行与停止

Launch or resume in PowerShell:

在 PowerShell 中启动或恢复：

```powershell
Set-Location 'C:\Users\fangz\Documents\Summer-Hand-intent-decoder-proj'
& '.\indy_loco\experiment\phase18_large_mingru\tuning\start_tuning.ps1'
```

The launcher refuses to overlap Phase17/Phase18 Python processes or bypass this
run's `STOP_AFTER_FOLD`. It uses the repository interpreter, explicit working
directory, hidden background launch, and unique stdout/stderr logs. It resumes
only when its own saved config exists. Outputs are under
`indy_loco/experiment/phase18_large_mingru/results/regularization_v2`.

启动脚本拒绝与现有 Phase17/Phase18 Python 进程重叠，也不会绕过本轮的
`STOP_AFTER_FOLD`。它使用仓库虚拟环境、明确工作目录、隐藏后台进程和唯一
stdout/stderr 日志；仅在本轮配置文件已存在时恢复。输出目录为
`indy_loco/experiment/phase18_large_mingru/results/regularization_v2`。

Inspect `progress.json`, epoch logs, checkpoints, and finally `metrics.json` and
`REPORT.md`. Successful completion requires verified complete metrics and all
30 selected test folds, not simply process exit. To stop after the current fit
is saved, create this run's marker:

可查看 `progress.json`、epoch 日志、检查点，以及最终的 `metrics.json` 和
`REPORT.md`。成功完成必须有完整且经过校验的指标及全部 30 个优胜配置测试折，
不能仅根据进程退出判断。若需在当前一次训练保存后停止，创建本轮标记：

```powershell
New-Item -ItemType File -Force -Path '.\indy_loco\experiment\phase18_large_mingru\results\regularization_v2\STOP_AFTER_FOLD'
```

Only remove that marker when intentionally resuming this run. Saved configuration,
code, environment, input, preprocessing, and checkpoint identities are verified.
Completed fits are reused; an interrupted incomplete fit restarts. Do not change
fingerprinted Python code after launch.

仅在决定恢复本轮训练时删除该标记。恢复时会核验配置、代码、环境、输入、
预处理及检查点身份；复用已完成训练，被中断且未完成的一次训练会重新开始。
启动后不要修改纳入指纹校验的 Python 文件。

## Interpretation / 结果解释

Report validation and final test R² separately. Historical tests have already
been exposed, overlapping folds and global hyperparameter tuning are not nested
cross-validation, and six sessions are not 30 independent recordings. These
results compare configurations within the existing benchmark; they are not an
independent unbiased test estimate. Historical Midsize was warm-started while
this experiment trains from scratch. A single seed does not establish seed
robustness, and the screening stage may miss a configuration that would win
across all folds.

验证集与最终测试集 R² 分开汇报。历史测试结果已经公开，各折有重叠，全局
超参数调优不是嵌套交叉验证，六个 session 也不等于 30 次独立记录。
这些结果用于现有 benchmark 内的配置比较，不是独立无偏的测试估计。
历史 Midsize 使用过 warm start，而本实验从头训练；单一 seed 不能证明
随机种子稳定性，筛选阶段也可能漏掉完整五折下更好的配置。

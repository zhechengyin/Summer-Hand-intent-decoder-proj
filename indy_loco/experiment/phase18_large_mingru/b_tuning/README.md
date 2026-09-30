# Final local tuning of model B / B 模型最后一轮局部调参

This Phase18 extension keeps the existing B architecture: width 128, three
minGRU blocks, causal temporal frontend, FFN expansion 2, and a 128 → 256 → 2
head. It has 374,402 parameters and 1,497,608 FP32 weight bytes (1.50 decimal
MB). Activations and workspace are additional. No new larger model is trained.

本轮作为 Phase18 的扩展，固定现有 B 架构：宽度 128、三个 minGRU block、
因果时序前端、FFN 扩展倍数 2，以及 128 → 256 → 2 输出头。参数量为
374,402，FP32 权重占 1,497,608 字节（十进制约 1.50 MB），不含激活与
工作区。本轮不训练更大的新架构。

## Prespecified recipes / 预先确定的配置

| Recipe / 配置 | Learning rate / 学习率 | Core weight decay / 核心权重衰减 | Head weight decay / 输出头权重衰减 |
|---|---:|---:|---:|
| B0: cached baseline / 复用基线 | 0.0010 | 0.010 | 0.030 |
| b1 | 0.0008 | 0.010 | 0.030 |
| b2 | 0.0012 | 0.010 | 0.030 |
| b3 | 0.0010 | 0.003 | 0.030 |
| b4 | 0.0010 | 0.030 | 0.030 |
| b5 | 0.0010 | 0.010 | 0.010 |

All other settings remain fixed: scratch initialization, seed 43, FP32,
AdamW, batch 128, gradient clipping 1.0, 60-epoch cosine schedule, at most
60 epochs, channel/stem dropout 0.1, and zero head dropout. The original
construction order and RNG reseeding are retained, making cached B0 a matched
baseline. Existing code, checkpoints, and result folders remain intact.

其他设置保持一致：从头初始化、seed 43、FP32、AdamW、batch 128、梯度
裁剪 1.0、60 epoch 余弦调度、最多 60 epoch、通道与 stem dropout 0.1、
输出头 dropout 为零。保留原模型构造顺序与随机种子重置，以确保可与缓存的
B0 对照。已有代码、检查点与结果目录保留。

EMA decay 0.99 is fixed for configuration selection and final testing. Raw
weights are retained as diagnostics. Each policy keeps its best checkpoint
by validation normalized MSE. Original early stopping is unchanged: 15 epochs
without either raw or EMA improving its own best validation loss. Validation
R² ranks the resulting EMA checkpoints; test results never select a recipe.

配置选择与最终测试统一采用 decay 0.99 的 EMA，raw 权重仅作为诊断。
两种权重分别按 validation 标准化 MSE 保存最佳检查点。提前停止规则保持
原样：连续 15 个 epoch 内 raw 与 EMA 均未改善各自最佳验证损失则停止。
随后按 EMA 检查点的 validation R² 排序，不使用 test 选择配置。

## Automatic sequence / 自动执行顺序

1. Verify immutable source/data fingerprints, all 30 frozen preprocessing and
   split receipts, cached B0 checkpoints, parameter counts, finite gradients,
   and final-only versus sequential/reference arithmetic.
2. Screen b1–b5 on fold 1 of all six sessions: 30 new fits. Include cached B0
   validation as a reference and select the best two **new** recipes.
3. Complete folds 2–5 for both finalists: 48 more fits, 78 new fits total.
4. Compare all 30 validation folds for B0 and both finalists. Freeze the winner;
   retain B0 on an exact tie. Test remains closed until every scheduled fit and
   all 90 finalist/baseline training receipts are verified.
5. Evaluate both new finalists on test (60 evaluations), verify/reuse the 30 B0
   test results, and report paired fold differences, mean/sample SD, per-session
   scores, learning curves, and unchanged model capacity.

1. 核验不可变源码/数据指纹、30 折预处理与划分、B0 检查点、参数量、有限
   梯度，以及仅最终输出与顺序递推/完整参考计算的一致性。
2. b1–b5 在六个 session 的第 1 折筛选，共 30 次新训练。缓存 B0 的
   validation 作为参照，选出最好的两个**新配置**。
3. 两个入围配置分别补齐第 2–5 折，再训练 48 次，总计 78 次新训练。
4. 用 B0 与两个入围配置的完整 30 折 validation 冻结胜出配置；完全持平时
   保留 B0。全部训练及 90 份基线/入围配置训练记录核验通过后才开放 test。
5. 对两个新配置执行共 60 次 test 评估，核验并复用 B0 的 30 次 test，
   汇报逐折配对差值、均值/样本标准差、各 session 结果、学习曲线与模型容量。

The six sessions, fivefold split, training-only normalization, 40-ms bins,
192 features, and independent zero-state 50-bin windows are unchanged. There
is no persistent streaming. Historical test results have already been exposed;
this is not nested cross-validation or a new untouched test set. Fold results
are correlated. Historical Midsize used warm-starting.

六个 session、五折划分、仅训练集拟合的归一化、40 ms bin、192 维特征及
每窗口独立零初始状态的 50-bin 输入均保持一致，不采用持续 streaming。
历史 test 结果此前已查看，因此本轮不是嵌套交叉验证，也不是新的未见 test
集。各折结果有关联，历史 Midsize 采用了 warm-start。

## Running and resuming / 启动与续跑

From the repository root, run the zero-optimizer-step preflight first:

从仓库根目录先运行零优化步骤的预检：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u indy_loco\experiment\phase18_large_mingru\b_tuning\train.py --device cuda --threads 4 --preflight-only
```

After it passes, launch the complete background sequence:

通过后启动完整后台队列：

```powershell
& .\indy_loco\experiment\phase18_large_mingru\b_tuning\start_tuning.ps1
```

The launcher automatically adds `--resume` when a config exists. It refuses
overlapping Phase17/18 Python processes; the runner also holds shared OS locks.
Unique stdout/stderr and launch records are under `../results/b_tuning_v1/logs`.
To request a stop after the current fold, create an empty `STOP_AFTER_FOLD` in
`../results/b_tuning_v1`. Inspect any failure before resuming; never remove
fingerprints or alter frozen code to bypass a mismatch.

存在配置文件时，启动脚本会自动添加 `--resume`。脚本拒绝与其他 Phase17/18
Python 进程重叠，训练程序同时持有共享系统锁。独立 stdout/stderr 与启动
记录保存在 `../results/b_tuning_v1/logs`。如需在当前折结束后停止，在
`../results/b_tuning_v1` 创建空文件 `STOP_AFTER_FOLD`。遇到失败先检查
原因，不通过删除指纹或修改冻结源码来绕过不一致。

FP32 deployment feasibility is evaluated separately under `../b_export` and
`../results/b_export_v1`, using the predetermined first session/fold EMA
checkpoint. This establishes export parity and toolchain compatibility;
actual board latency and SDRAM placement require later hardware measurement.

FP32 部署可行性在 `../b_export` 和 `../results/b_export_v1` 单独验证，
使用预先确定的第一个 session/第 1 折 EMA 检查点。该验证确认导出数值一致性
与工具链兼容性，板端延迟与 SDRAM 放置仍需后续硬件实测。

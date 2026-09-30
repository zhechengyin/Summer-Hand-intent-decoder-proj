# Phase18 large minGRU / 大容量 minGRU

Train a width-128 minGRU with three residual blocks, FFN expansion 2, and a
128 → 768 → 1024 → 2 output head with GELU hidden activations. It has
**1,211,906 parameters and 4,847,624 bytes of FP32 weights (4.847624 MB)**.
Each prediction retains the frozen 50-bin, 40-ms/bin input window and resets
the recurrent state to zero. No hidden state is carried between windows.

训练宽度 128、三个残差 block、FFN 扩展倍数 2 的 minGRU，输出头为
128 → 768 → 1024 → 2，隐藏层使用 GELU。模型共有 **1,211,906 个参数，
FP32 权重占 4,847,624 字节（4.847624 MB）**。每次预测沿用固定的
50-bin 窗口，每个 bin 为 40 ms；循环状态从零开始，不跨窗口传递。

The final block computes its FFN only at the final timestep. Final normalization
and the output head also run only once. Earlier recurrent updates and the first
two blocks' FFNs are retained because they affect the final prediction. The
model returns `[batch, 1, 2]`, preserving the frozen evaluator's `[:, -1]`
interface. The estimated dense computation is **13,649,920 MACs per 50-bin
window**, excluding normalization, nonlinearities, elementwise operations,
and memory movement.

最后一个 block 的 FFN、最终归一化和输出头仅计算最后一个时间步。
前面的循环状态更新及前两个 block 的 FFN 会影响最终预测，因此保留。
输出形状为 `[batch, 1, 2]`，兼容原评估函数的 `[:, -1]` 接口。
每个 50-bin 窗口估算为 **13,649,920 次密集 MAC**，不包含归一化、
非线性函数、逐元素运算和内存搬运。

Training uses the parallel minGRU formulation. A sequential arithmetic path
implements the same zero-initialized recurrence for verification and future
deployment; this does not require separate weights or persistent streaming.
Batched projections apply the same weights across the window instead of
launching a separate projection for every timestep. This expresses weight reuse
in software, but does not implement STM32 SDRAM/DMA transfers or establish
measured board latency.

训练采用并行 minGRU 公式，并提供使用同一组权重、相同零初始状态的顺序
递推路径，用于等价性验证和后续部署；不需要单独训练顺序版本。
投影运算把整个窗口合并计算，使同一组权重用于多个时间步。这是软件层面的
权重复用安排，尚未实现 STM32 SDRAM/DMA 搬运，也不代表已验证板端延迟。

## Frozen data and selection / 固定数据与选择规则

The runner reuses and verifies the existing preprocessing, six sessions, and
five folds. It checks canonical source and GUI arrays, channel selection,
calibration, target scalers, split indices, and preprocessing fingerprints.
Phase17 code, results, checkpoints, and its stopped sweep remain unchanged.
Models initialize from scratch; only the previous winning hyperparameters
provide a search anchor.

脚本复用并校验原来的预处理、六个 session 和五折划分，核验源数据与 GUI
数组、通道选择、校准、目标标准化、划分索引和预处理指纹。Phase17 的代码、
结果、检查点及已停止的搜索保持原状。新模型从头初始化，仅将之前胜出的
超参数作为搜索起点。

All trials use seed 43, FP32, AdamW, cosine learning-rate decay over 60 epochs,
up to 60 epochs with patience 15, batch size 128, gradient clipping at 1.0,
and equal learning rates for the stem and the remaining model. Within each
fit, retain the checkpoint with minimum validation normalized MSE. The six
predeclared configurations are:

所有配置使用 seed 43、FP32、AdamW、周期为 60 epoch 的余弦学习率衰减、
最多 60 epoch、patience 15、batch size 128、梯度裁剪 1.0；stem 与其他
部分学习率一致。每次训练保存验证集标准化 MSE 最低的检查点。六组预设配置为：

| Trial / 配置 | Learning rate / 学习率 | Weight decay | Channel dropout / 通道 dropout | Dropout |
|---|---:|---:|---:|---:|
| t00: previous winner anchor / 原优胜参数 | 0.001 | 0.01 | 0.1 | 0.1 |
| t01 | 0.0006 | 0.01 | 0.1 | 0.1 |
| t02 | 0.0003 | 0.01 | 0.1 | 0.1 |
| t03 | 0.0006 | 0.03 | 0.2 | 0.2 |
| t04 | 0.0006 | 0.003 | 0.05 | 0.05 |
| t05 | 0.001 | 0.03 | 0.2 | 0.2 |

Screen all six configurations on fold 1 of every session: **36 fits**. Rank
by mean validation R² and promote two configurations. Complete folds 2–5 for
those two: **48 additional fits, 84 total**. Select and freeze the winner using
its complete 30-fold validation results, then evaluate only its 30 test folds.
No test metrics select hyperparameters, and no seed-44/45 extension is planned.

先在每个 session 的第 1 折筛选全部六组配置，共 **36 次训练**；按平均验证集
R² 选出两组，再完成它们的第 2～5 折，共新增 **48 次，总计 84 次训练**。
根据完整 30 折验证结果选定并冻结优胜配置后，只评估该配置的 30 个测试折。
测试指标不参与超参数选择，本轮不扩展 seed 44/45。

## Start, inspect, and stop / 启动、查看与停止

From PowerShell, launch or resume in the background:

在 PowerShell 中执行以下命令，可后台启动或恢复训练：

```powershell
Set-Location 'C:\Users\fangz\Documents\Summer-Hand-intent-decoder-proj'
& '.\indy_loco\experiment\phase18_large_mingru\start_training.ps1'
```

The launcher checks for active Phase17/Phase18 Python processes, uses the
repository virtual environment and explicit working directory, and starts a
hidden process with unique stdout/stderr logs. It adds `--resume` only when
the new run's `config.json` exists. Process enumeration must succeed; otherwise
it refuses to launch. The Python runner also holds OS locks. The printed launch
receipt contains the launcher PID and log paths; the active training PID is
recorded in `progress.json`.

启动脚本先检查是否有 Phase17/Phase18 Python 进程，使用仓库虚拟环境和明确
工作目录，以隐藏窗口启动，并生成唯一的 stdout/stderr 日志。仅当新实验的
`config.json` 已存在时添加 `--resume`。如果无法枚举进程，则拒绝启动；
Python 主程序还会持有操作系统锁。启动记录包含 launcher PID 和日志路径，
实际训练 PID 记录在 `progress.json` 中。

Outputs are under `indy_loco/experiment/phase18_large_mingru/results/large_mingru_v1`.
Inspect `progress.json`, `logs/`, per-fit epoch logs and checkpoints, and later
`final_selection.json`, `metrics.json`, and `REPORT.md`. A process exit alone
does not prove completion; verify the final complete metrics and all 30 test
folds. Resume verifies configuration, code, environment, data, and checkpoint
identity and reuses completed fits. An interrupted incomplete fit restarts.
Do not modify fingerprinted training files after launch.

输出位于 `indy_loco/experiment/phase18_large_mingru/results/large_mingru_v1`。
可查看 `progress.json`、`logs/`、各次训练的 epoch 日志和检查点，以及后续的
`final_selection.json`、`metrics.json` 和 `REPORT.md`。进程退出不等于成功完成，
应确认最终指标标记为完成且含全部 30 个测试折。恢复时会核验配置、代码、环境、
数据及检查点身份并复用已完成训练；被中断且未完成的一次训练会重新开始。
启动后不要修改纳入指纹校验的训练文件。

To stop after the current fit has saved, create this experiment's stop marker:

若需在当前一次训练保存后停止，创建本实验的停止标记：

```powershell
New-Item -ItemType File -Force -Path '.\indy_loco\experiment\phase18_large_mingru\results\large_mingru_v1\STOP_AFTER_FOLD'
```

Remove that marker only when intentionally resuming this experiment. Leave
Phase17's separate stop marker in place.

仅在决定恢复本实验时删除上述标记。保留 Phase17 独立的停止标记。

## Interpretation / 结果解释

Report test R² mean and sample standard deviation over the 30 verified folds,
paired differences versus historical Midsize, parameter count, and FP32 weight
size. Historical Midsize was warm-started while this model trains from scratch.
Historical test results have already been exposed, folds overlap in training
data, and global hyperparameter selection is not nested cross-validation.
Consequently, these are comparative results on the existing benchmark, not an
independent unbiased test estimate. The six sessions also do not provide 30
independent recording replicates. A larger model is not guaranteed to improve
R², and GPU runtime or MAC counts do not establish MCU latency.

最终汇报 30 个已验证测试折的 R² 均值与样本标准差、相对历史 Midsize 的配对
差值、参数量和 FP32 权重大小。历史 Midsize 使用过 warm start，本模型从头
训练。历史测试结果已经公开，各折训练数据有重叠，全局超参数选择也不是嵌套
交叉验证。因此这些结果用于现有 benchmark 内的比较，不能视为独立无偏的
测试估计；六个 session 也不等于 30 次独立记录。更大模型不保证提高 R²，
GPU 耗时和 MAC 数量不能证明 MCU 的实际延迟。

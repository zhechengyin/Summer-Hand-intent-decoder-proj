# Phase 16 — modest parameter scaling with fast stopping

按最新要求，将默认模型从 256 宽度、6 层 TCN + 3 层 GRU，缩小为 **80 宽度、
4 层 TCN + 1 层 GRU**；启用同 session / 同 fold 的最终 Midsize 权重迁移，
并增加更快的验证集提前停止判据。只交付脚本与无训练验证，由用户运行训练。

## 默认配置

| 项目 | 原 Midsize | 当前 Phase 16 |
| --- | --- | --- |
| Encoder width | 64 | 80 |
| TCN kernel / dilation | 3 / 1,2,4,8 | 不变 |
| TCN residual blocks | 4 | 4 |
| GRU hidden / layers | 64 / 1 | 80 / 1 |
| 输入 / 输出 | 192×50 / 50×2 | 不变，loss 取 timestep 49 |
| 参数量 | 86,978 | 131,762（1.515 倍） |
| FP32 权重 | 347,912 B | 527,048 B（0.503 MiB） |

当前 encoder 参数 92,720，decoder（GRU + head）参数 39,042。相对前一个
3,202,306 参数的默认版本减少约 96% 参数；这不是承诺训练耗时同比下降 96%。
TCN 感受野仍为 31 bins，输入窗口仍为 50 bins。权重字节数不包括 activation、
工作区、对齐和固件，因此不能当作总运行内存或 external-RAM 放置验证。

## 权重迁移是什么

旧模型已经通过训练学到了输入与输出的关系。权重迁移把这些已学到的参数作为
新模型的起点，而不是把所有参数重新随机初始化。

这里每个新 fold 只用同一 session、同一 fold 的最终 Phase-13 checkpoint，
按 manifest 校验 SHA、fold、通道和预处理，不挑分数最高的 test fold，也不跨 fold
借用权重。默认可以复制全部 86,978 个旧参数；44,784 个新增参数保持 PyTorch 默认
随机初始化，seed 仍为 43。迁移之后所有参数继续训练，没有冻结旧层。

`transfer.py` 对 GRU 的 reset/update/new 三个 gate 分别映射到新 hidden 宽度，
避免直接复制前 192 行造成 gate 错位；卷积按 causal lag 对齐。如果显式修改 kernel
或 dilation，无法对应到相同 lag 的旧 tap 不迁移，并记录数量。新增层保持随机初始化。

这是一种**部分热启动**，不是保证函数不变的扩容：新连接和更宽的 LayerNorm 会改变
初始输出。因此可能减少达到良好验证结果所需的 epoch，但不保证一定更快收敛或更高分。
迁移不降低单个 epoch 的计算量；单轮提速主要来自更小的模型。没有执行训练来声称实测提速。

## 新 criterion：更快停止，不修改 loss

- Loss 仍是 timestep 49 的 normalized MSE。
- 至少训练 **4 epochs**。
- 如果连续 **3 epochs** 相对上一个显著改善点的验证 normalized MSE，累计下降
  未达到 **0.5%**，就停止当前 fold。达到阈值时重置计数。
- 最多仍为 **20 epochs**，CosineAnnealingLR 的 T_max 仍为 20。
- checkpoint 始终保存 **实际验证 MSE 最低**的 epoch；即使改善不足 0.5%，也会
  更新最佳 checkpoint，只是不一定重置停止计数。
- 只用 validation 决定停止。保存最佳 checkpoint 之后才评估 test。

例：验证 MSE 为 1.000、0.998、0.996、0.994，累计改善超过 0.5%，重置计数；
若始终为 1.000，第四个 epoch 后停止。此判据可能更早停止，但可能放弃后期缓慢的提升，
是用户为加快完成实验而明确要求的取舍，不再宣称提前停止规则与 Phase 13 完全相同。

## 仍然锁定的部分

- 六个 session × 五折，fold/训练 seed 43；原 reach eligibility、划分与训练通道排序。
- 连续 EWMA alpha 0.1、40 ms bins、7 分钟/10,500 bins 无标签校准、50-bin past-only 窗口。
- std floor 的 train-only 60 s blocks、percentile/fallback；target scaling 只来自训练 bins。
- count/EWMA 配对 dropout 0.2、模型 dropout 0.1、GRU inter-layer dropout 0。
- AdamW，GRU/head LR `3e-4`、encoder LR `7.5e-5`、weight decay 0.025。
- batch 128、gradient clipping 1.0、全参数更新、相同 scheduler、相同 validation best 选择。
- 不自动缩小 batch，不使用 AMP、梯度累积或新的数据增强。
- 默认 CPU / 4 threads；可显式指定 `--device cuda/mps`，后端写入配置，不宣称跨设备逐位一致。

`protocol.py` / `session_data.py` / `protocol_lock.json` 保留原始冻结实现与哈希。
`fast_training.py` 使用相同 optimizer/epoch 主体，仅替换最后的停止条件；启动时对比 AST，
防止其他训练语句意外漂移。`stopping.py` 实现新停止判据。CLI 不开放其他训练超参数。

每个 fold 的通道、calibration mean/std、std floor 和 target scaling 与已有 Midsize
逐元素校验，reach/bin 数量保持一致，并保存实际 split/特征数组的 SHA-256。
校准输入前缀允许包含分到任意 fold 的 inputs，保持原 Midsize 既有无标签校准规则。

## 运行

从 `Summer-Hand-intent-decoder-proj` 根目录执行；环境需要 PyTorch、NumPy、h5py。

```bash
# 只做迁移/shape/元数据预检，不训练
.venv-deploy/bin/python indy_loco/experiment/phase16_parameter_scaling/run.py --validate-only

# 再检查全部 30 folds 的实际预处理，不训练
.venv-deploy/bin/python indy_loco/experiment/phase16_parameter_scaling/run.py --dry-run

# 开始默认 80-width + 权重迁移 + 新停止判据训练
.venv-deploy/bin/python indy_loco/experiment/phase16_parameter_scaling/run.py

# 只恢复当前新版本的同配置运行
.venv-deploy/bin/python indy_loco/experiment/phase16_parameter_scaling/run.py --resume
```

默认新输出：`results/fast_transfer_scaled/`。旧 `results/scratch_scaled/` 的训练结果
保持不动，不能用新架构和新代码 resume 旧运行。若你在其他终端仍运行旧版本，需要在那里
先结束旧进程再启动新命令；修改文件不会改变已经运行中的 Python 模型。

只选一个 session/fold 时，不降低 epoch/batch 等设置；结果只是子集：

```bash
.venv-deploy/bin/python indy_loco/experiment/phase16_parameter_scaling/run.py \
  --session indy_20160622_01 --fold 1 --output-name fast_indy_fold1
```

需要稍大一点可显式设置宽度，层数仍保持默认，不会再自动变成 256：

```bash
.venv-deploy/bin/python indy_loco/experiment/phase16_parameter_scaling/run.py \
  --encoder-width 96 --decoder-hidden-size 96 --output-name fast_width96
```

架构选项仍包括 encoder kernel/dilations/layers、decoder hidden/layers。
`--encoder-layers` 必须与 dilation 数量一致。每种配置使用独立 output-name。

为了比较架构效果，原 64-width 架构也可从对应最终 Midsize checkpoint 继续训练，
使用同一快速 criterion：

```bash
.venv-deploy/bin/python indy_loco/experiment/phase16_parameter_scaling/run.py --model baseline
```

它写入 `results/fast_transfer_baseline/`。相对历史 Phase-13 分数的变化同时包含额外训练
和新停止策略，不能全部归因于参数量；不要反复按 test 结果筛选架构。

## 输出与验证

输出仍包含 config/state、checkpoints、folds/epochs/summary CSV 和 metrics JSON。
迁移来源路径、SHA、复制比例/未映射参数数、架构和停止策略都会记录。Resume 核对配置、
代码、环境、数据和 checkpoint 哈希；跳过已完成 fold，未完成 fold 从同一迁移起点重跑。
不会写回原模型包、自动 promotion，或更新 GUI/固件。

本次未训练验证：18 项 unittest、30 folds 的迁移/有限输出/预处理检查均通过；
Ruff lint/format 通过。没有启动正式训练或生成新的 fast-transfer 训练 checkpoint。

```bash
.venv-deploy/bin/python -m unittest indy_loco.experiment.phase16_parameter_scaling.test_contract -v
```

现有 Cube.AI graph/权重尺寸固定，新的模型仍需另行转换和 external-RAM 部署适配，
不能直接上传到现有 runtime，也没有验证 40 ms 板上时限。

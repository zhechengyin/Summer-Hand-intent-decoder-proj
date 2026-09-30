# Phase 17 — 三种时序架构与 Midsize 的固定协议比较

状态（2026-09-26）：**三个模型均已完成 30 folds，checkpoint/prediction 指纹验证通过。**
Test R²：minGRU **0.6988 ± 0.0750**，Mamba-2 **0.6865 ± 0.0861**，
Transformer **0.6818 ± 0.0818**（fold mean ± sample SD）。
用户已授权执行独立的 [minGRU/Mamba-2 sweep](sweep/README.md)；
原始 Python 文件和结果保持冻结。以下内容记录原始固定配方，新的调参预算另见 sweep 协议。
本阶段比较 minGRU、Mamba-2、轻量 causal Transformer，沿用当前 Midsize
完整的数据处理和固定五折协议。三者替换原 TCN+GRU 主干，不引入新输入特征。

## 架构与容量

共同路径：`192×50 → 转为50×192 → Linear(192,64) → 时序主干 → LayerNorm64
→ Linear(64,2)`，仅 timestep 49 参与 loss 和最终预测。保留原 count/EWMA
配对 channel dropout 0.2；在输入投影之后、时序主干之前使用一次 dropout 0.1。
各候选内部不额外叠加 dropout；norm 只沿 feature 维，不跨时间统计。

| 模型 | 提议的时序主干 | 参数量 | 相对 Midsize | FP32 权重 |
|---|---|---:|---:|---:|
| 当前 Midsize | TCN64×4，k3/d1,2,4,8 → GRU64×1 | 86,978 | 1.000× | 339.76 KiB |
| minGRU | 3个 residual minGRU64 block，每块 FFN64→128→64 | 88,066 | 1.013× | 344.01 KiB |
| Mamba-2 | 2个 residual Mamba-2 block，d_model64 / expand2 / state64 | 82,290 | 0.946× | 321.45 KiB |
| Causal Transformer | 2个 pre-LN block，d64 / 4 heads / FFN64→160→64 | 87,810 | 1.010× | 343.01 KiB |

候选参数已经用实际网络实例核对，与以下逐层计算值一致。
运行 `python indy_loco/experiment/phase17_architecture_comparison/capacity.py`
可重算，脚本仅依赖标准库，没有训练入口。权重容量不等于总 RAM。

**minGRU：优先实现。** 每块为 `x + minGRU(LN(x))`，再接
`x + FFN(LN(x))`；FFN 使用 GELU。minGRU 使用带 bias 的64→128投影，拆为
candidate/update gate；不增加 output projection。选论文附录B的正值 candidate
版本（x≥0时g(x)=x+0.5，否则sigmoid(x)），hidden 初值为0。单块参数
8,320 + 16,576 + 256 = 25,152；共同投影/最终norm/head为12,610。
它测试简化门控和更深残差结构能否在相近容量下改善泛化。论文依据是
[Were RNNs All We Needed? (2024)](https://arxiv.org/abs/2410.01201)；
这里的层数和宽度是本项目适配，未预先证明其在 Indy/Loco 上优于 GRU。

**Mamba-2：测试选择性状态空间。** 每块为 `x + Mamba2(LN(x))`；
`d_model=64, expand=2, d_inner=128, d_state=64, headdim=16,
nheads=8, ngroups=1, d_conv=4`。输入/输出 projection 无 bias，depthwise conv
有 bias；gated RMSNorm scale128、dt/A/D向量各8，`D_has_hdim=False`，
无额外 FFN、无 learned initial state。每块核心34,712参数，外层LN128，
两块加共同层为82,290。保留官方因果卷积、SiLU、dt初始化和稳定状态参数化。
出处为 [Mamba-2 (2024)](https://arxiv.org/abs/2405.21060) 和
[作者实现](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)。
[POSSM (2025)](https://arxiv.org/abs/2506.05320) 为 SSM 用于实时 intracortical
decoding 提供任务动机；本方案不采用它的 individual-spike tokenizer 或预训练，
以保持本项目原有 preprocessing。当前实现使用纯 PyTorch 的 dense SSD，对固定
50步窗口计算与zero-state recurrence等价的输出，不依赖mamba-ssm、Triton或自定义CUDA。
已核对递推输出和梯度，Windows CPU/CUDA前向/反向均通过。该实现计算复杂度为O(T²)，
不宣称具有官方长序列优化kernel的线性复杂度或速度，也尚未验证板上部署。

**Causal Transformer：测试窗口内直接 attention。** 每块使用两个 affine
LayerNorm，带bias的Q/K/V/output projection，4个16维head，GELU FFN160。
单块16,640 + 20,704 + 256 = 37,600参数。位置编码为固定 sinusoidal，
不增加参数；attention使用包含对角线的严格因果 mask。每个bin是一个token，
不做patching/downsampling，不做masked-spike预训练。Transformer并非2026年的新发明，
但它是本仓库的新对照家族；选择它是为了区别递归压缩与窗口内直接检索。
依据 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，神经任务动机参考
[NDT2 (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf)。
这里不是复现完整 NDT2，不能引用其预训练结果作为本模型预期精度。

三者每次都重新处理同一个50-bin窗口；minGRU/Mamba状态在窗口开始清零，
Transformer不跨窗口携带KV。模型显式输入仍为2 s；其中EWMA本身仍按原协议
含有更早输入的因果历史。不能把跨窗口持久状态混入本轮比较。

除权重外，理论持久状态也不同：minGRU为3×64 floats（0.75 KiB）；
Mamba-2的两层SSM+conv状态为18,432 floats（72 KiB）；Transformer若实现50-bin
KV缓存为12,800 floats（50 KiB）。这些不是本轮窗口推理的实测峰值RAM，不能与
权重直接相加作为部署内存。实际 activation、峰值RAM、MACs和latency留到实现后实测。

## 固定数据合同

以当前 Phase-13 `final_30fold` 为比较基准，复用 Phase-16 中已冻结的
`session_data.py`、`protocol.py` 的数据函数和 `protocol_lock.json` 身份信息。
实现应使用 active 代码，不导入 history，也不调用 Phase-16 transfer/fast-stop。

| 项目 | 必须保持的行为 |
|---|---|
| Sessions | indy_20160622_01、indy_20160630_01、indy_20170131_02；loco_20170210_03、loco_20170215_02、loco_20170301_05 |
| Bin | 原4 ms spike presence按10 samples累加成40 ms counts；velocity同区间均值 |
| Channels | 96；Loco仅用training reaches做既有通道排序/选择，保持原顺序 |
| Features | 96 counts + 96连续causal EWMA，alpha=0.1，不在reach边界重置 |
| Calibration | session最初7 min=10,500 bins无标签输入，原mean/std/floor规则 |
| Std floor | train reaches按时间拼接后的60 s blocks，10th percentile及原fallback |
| Target scaling | 仅训练target bins拟合mean/std |
| Window/target | 截至当前bin的50 bins，最后一步49，normalized MSE，无future lookahead |
| Reaches | ≤8 s且有完整bin，原eligibility与聚合规则 |
| Split | seed43打乱eligible reaches成5份；每折4份训练，剩余份交替分val/test，约80/10/10 |
| First eligible bin | 0-based 10499，保持原 `max(CALIBRATION_BINS-1,WINDOW_BINS-1)` |

这是沿用部署式 session 协议：无标签 calibration prefix 可含分属任何fold的inputs，
连续EWMA/window也会跨reach/split边界。不将其改成reach-local，也不声称所有输入
预处理仅使用训练fold。训练前对channels和split counts精确核对，保存实际reach/bin
indices、特征/velocity数组SHA-256。Windows/x86环境重算部分FP32统计与历史checkpoint
有约1e-7差异：先以rtol=5e-7、atol=5e-8检查原函数重算值，超过容差直接失败；
随后使用同session/fold冻结checkpoint的calibration/floor/target scaling，并重新归一化。
最终使用的scalers逐元素等于历史值，重算偏差写入`preprocessing_evidence.numeric_reproducibility`。
不声称跨平台连续EWMA中间运算逐位一致；三个新模型使用相同代码和数组指纹。

## 超参数和公平对照

第一轮**冻结已选超参数，不新增按模型单独调参的搜索**；每个fold仍只用
validation normalized MSE选checkpoint。架构宽度/深度按容量预先设定，不按test改。
这是固定训练配方下的比较，不代表每个架构经过各自最优HPO。

| 项目 | Phase17建议值（来自Phase13最终训练配方） |
|---|---|
| Optimizer | AdamW，weight decay0.025；原betas/eps |
| LR | encoder/stem7.5e-5，temporal decoder/head3e-4 |
| Epoch/batch | 最多20epochs，batch128，gradient clipping1.0 |
| Scheduler | CosineAnnealingLR，T_max20 |
| Selection/stop | 实际最低validation normalized MSE；原patience6 |
| Seed | fold/training43，与历史一致 |
| Regularization | 配对channel dropout0.2，model dropout0.1；无新数据增强 |
| Precision | FP32；所有模型同一训练device，记录版本/backend；不自动减batch或启用AMP |

新架构必须显式列出optimizer参数组：新模型的stem为低LR，全部时序blocks、最终norm
和head为高LR；Midsize control仍使用原spatial+TCN低LR、GRU+head高LR。
这是新架构的预先声明映射，不能直接按旧 `gru.`/`head.` 名称筛选，否则新block会
错误进入低LR组。保留原全参数weight decay规则，不默默应用Mamba仓库专用的
no-weight-decay分组；SSM特定初始化属于架构实现，必须记录。固定配方的效果需实测。
新模型的时序主干大部分使用高LR，而Midsize的TCN仍使用低LR；因此scratch对照
消除了warm-start/额外训练预算的差异，但结果仍是预先声明配方下的整体模型比较，
不能称为严格只改变网络拓扑的消融。

当前历史 Midsize 是 same-fold Phase7 warm start 后再做Phase13训练，不能直接把
三个scratch模型与它的差异全归因于架构。建议最终同时保留两个参照：

1. **历史 Midsize**：冻结的30fold最终成绩，回答是否超过当前实际模型。
2. **Matched scratch Midsize**：按上表与三个候选一起从头训练，回答相同初始化
   来源、训练预算和选择规则下的模型差异。每种架构使用自己的已声明初始化；seed同为43。

因此正式运行建议是3×30=90个新模型fold，另加30个scratch Midsize control，共120 fits。
本次未启动其中任何fit。Phase16的warm start和min4/patience3/0.5%快速停止不混入。
后续若调整预算或做HPO，必须对四组采用同等validation-only选择规则，并记录协议修订。

## 结果汇报约定

已有历史 Midsize 数据来自
[`phase13_round3_summary.csv`](../phase13_deployment_validation/results/rolling_retrain/final_30fold/phase13_round3_summary.csv)：

| Session | 历史Midsize test R²（5fold mean ± sample SD） |
|---|---:|
| indy_20160622_01 | 0.8381 ± 0.0214 |
| indy_20160630_01 | 0.7152 ± 0.0172 |
| indy_20170131_02 | 0.7725 ± 0.0562 |
| loco_20170210_03 | 0.7163 ± 0.0450 |
| loco_20170215_02 | 0.6826 ± 0.0431 |
| loco_20170301_05 | 0.7221 ± 0.0638 |
| 全部30fold | **0.7411 ± 0.0656** |

每折先分别算x/y R²再平均；总体对30个fold等权平均、SD使用ddof=1。
Indy15fold为0.7753±0.0618，Loco15fold为0.7070±0.0510。
不用六个best-test-fold的0.7944作为主比较分数。

最终三模型和scratch control分别报告各session五折、总体和分subject test结果；
同时报告配对ΔR²（对历史和scratch各一组）、wins/30、最差fold/session、RMSE和
capacity（实际参数、FP32权重、计算量、同硬件峰值RAM和推理时间）。同一session的不同fold相互相关，
推断统计以6个session mean配对为主，不把30fold当独立样本。五折主要衡量本组
session内划分稳定性，不等于跨天、跨subject或多训练seed robustness。

全部候选设计/训练选择冻结后再统一打开test进行最终报告，不用test选epoch、挑fold、
改宽度或重新调超参数。历史7min校准来自已经查看过test的研究过程，因此本实验是
沿用固定benchmark的对照，不宣称整个研究过程保有从未暴露的test。

训练前实现检查包括：实际参数计数、输出shape/有限值、因果性、reference recurrence
及gradient一致性、30fold预处理/indices一致性、val-only选择和checkpoint身份记录。
本次16项无训练检查通过，包括minGRU/SSD递推输出和梯度、Mamba卷积及gate、因果性、
参数/LR分组、validation选择与patience、checkpoint完整性和两阶段resume。
真实六session×五折dry-run通过；三个CUDA入口和RTX4070 Laptop上的batch128前向/反向通过。
检查没有执行真实optimizer更新；候选test精度、完整训练耗时和板上兼容性仍待验证。

## 运行三个训练脚本

在Windows PowerShell中，从仓库根目录依次运行。每条命令独立训练一个模型的全部30fold。
本机已有Python3.14 / PyTorch2.11 CUDA环境；这里明确选同一CUDA后端。脚本默认是CPU，
也可给三个模型统一改成`--device cpu`。MPS选项保留，但本次未验证MPS。

```powershell
Set-Location C:\Users\fangz\Documents\Summer-Hand-intent-decoder-proj
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\train_mingru.py --device cuda
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\train_mamba2.py --device cuda
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\train_transformer.py --device cuda
```

三个入口固定架构，共用`run.py`、`training.py`及冻结数据函数；支持`--help`。
第一阶段保存全部指定fold的validation-selected checkpoints，第二阶段才计算test。
它们不会自动启动其他模型、更新部署模型包或修改历史成绩。

只做检查、断点继续或选择子集：

```powershell
# 完整30fold真实预处理检查；没有训练或test评价
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\train_mingru.py --dry-run

# 仅检查模型前向/反向、容量和原checkpoint元数据
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\train_mamba2.py --validate-only --device cuda

# 保留原命令的全部参数，追加--resume
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\train_mingru.py --device cuda --resume

# 子集运行使用独立输出名；子集不算完整30fold结果
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\train_transformer.py --device cuda --session indy_20160622_01 --fold 1 --output-name transformer_subset

# 可选：同预算scratch Midsize对照（不会被三个入口自动执行）
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\run.py --model midsize_control --device cuda
```

Resume验证配置、代码、Python/PyTorch/NumPy环境、原始输入文件、预处理数组和checkpoint
哈希。已保存完整checkpoint的fold不再训练；训练过程中中断、尚未保存最终checkpoint的
fold从seed43重新开始，并不恢复epoch内optimizer状态。已有test receipt和预测会核验后复用。
默认目录已存在时必须加`--resume`或用新的`--output-name`；不要在训练之间修改代码。
同输出目录有进程锁，避免两次命令同时写一个run。

数据位置：默认读取`indy_loco/data`；Indy可只读复用Phase13已保存的`*_4ms.npz`，
不要求再次下载MAT。缓存核对session/source-MD5元数据、完整reaches、GUI全时间counts/velocity；
没有原MAT时不会声称重算其原文件MD5。GUI路径自动识别相邻`BCI-STM32-Plot`或本机
`../STM32/BCI-STM32-Plot`。本机三个Loco processed NPZ已从现有Phase12 checkout补齐，
复制哈希一致且30fold验证通过。迁移到其他机器时可以显式使用`--data-root`、
`--indy-cache-root`、`--gui-root`；缺失或不一致时会报错，不会换数据或跳过验证。

## 输出与后续汇报

三个默认目录为`results/mingru/`、`results/mamba2/`、`results/transformer/`。
其中`epochs/`持续更新每fold的epoch日志；`checkpoints/`保存权重、归一化、实际split indices
和SHA；`predictions/`保存test bins、target和prediction；`fold_results/`保存可恢复结果。
test阶段生成`config.json`以外的`metrics.json`、`folds.csv`、`summary.csv`、`epochs.csv`。
报告自动包含历史Midsize配对ΔR²、每session/subject/总体统计、RMSE、实际参数量、FP32权重、
训练时间及test批量预测用时。批量预测用时不是板上单次延迟，峰值RAM字段保留null。

训练结束后可运行下列命令，它只读取保存结果，不重新预测test：

```powershell
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\summarize.py
```

输出`results/comparison/REPORT.md`、`models.csv`、`sessions.csv`和`comparison.json`。
默认要求三个模型都完成30fold，并核对跨模型的配方、环境、输入和fold一致性；可用
`--allow-partial`查看明确标记的部分结果，或`--run <directory>`指定自定义输出目录。
如果scratch control也完成，会加入其配对差值。完成后可直接让助手读取这些文件作最终汇报。

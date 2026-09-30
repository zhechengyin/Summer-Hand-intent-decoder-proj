# Phase18: equivalent minGRU CUDA / 等价 minGRU CUDA 对照

Scope is only candidate/gate computation, log-domain state scan and first-order
backward. Existing Linear projections, causal encoder, FFNs, normalization,
dropout, CPU window creation, transfers, AdamW, clipping, EMA and evaluation are
identical between arms. No mixed precision, TF32, activation approximation or
architecture change. Frozen prior experiment files are not edited.

仅替换 candidate/gate、log-domain 状态计算与一阶 backward。两组保留相同
Linear、encoder、FFN、归一化、dropout、CPU 窗口构造与传输、AdamW、梯度
裁剪、EMA 和评估。无混合精度、TF32、激活近似或架构修改。旧实验文件不修改。

The active **v3** candidate fuses positive-candidate/gate log-term preparation
and its backward into two CUDA kernels. It preserves PyTorch's native cumsum,
logcumsumexp/logsumexp, state finalization and their backward. It matches
softplus's threshold20 and multiply-then-divide backward evaluation order.
Existing projections and their gradients remain PyTorch. Eager, first-order
autograd only; no torch.compile integration is claimed.

当前 **v3** 只把 candidate/gate 的 log-term 准备及对应 backward 融合成
两个 CUDA kernel；状态 scan、最终状态计算与其反向传播保留 PyTorch。
保留 softplus 阈值20及其 backward 先乘后除的浮点顺序。Linear 投影及其
梯度仍由 PyTorch 执行。只支持 eager、一阶梯度，不声称支持 torch.compile。

Rejected candidates are preserved with source snapshots and exact errors:
**v1** fused the full log scan and used a mathematical reverse recurrence;
the amplified-input gradient comparison exceeded the preset FP32 threshold.
Diagnostic comparison to FP64 found that the original FP32 scan also incurs
cancellation error, so mathematical equivalence alone was insufficient for
the requested PyTorch-matching gate. **v2** retained the native scan but used
sigmoid-factorized softplus gradients; it passed output/gradient tolerance
checks but failed the AdamW single-update check near zero gradients. **v3**
matches the original derivative's operation order, without loosening any gate.
The full-fusion kernels remain in the file as rejected experimental code and
are not installed into the timed model.

失败候选及源码快照完整保留：**v1** 全 scan 融合在放大输入上未通过严格梯度
对照；FP64诊断显示原 FP32 scan 自身也有抵消误差，因此数学等价仍不足以
通过与原 PyTorch 的比较。**v2** 用 sigmoid 形式重写 softplus 导数，通过
输出/梯度检查但未通过接近零梯度时的 AdamW 单步更新检查。**v3** 改为原浮点
运算顺序，没有放宽阈值。被拒绝的全融合代码保留，但不会用于计时模型。

Use the existing torch-bundled NVRTC via ctypes; no nvcc/MSVC install is needed.
Kernel flags disable FMA contraction and do not enable fast math. CUDA kernels
launch on PyTorch's current stream and use PyTorch-owned FP32 allocations.

使用 PyTorch 自带 NVRTC，通过 ctypes 编译和调用，无需安装 nvcc/MSVC。
编译禁用 FMA 收缩，不启用 fast math。Kernel 使用 PyTorch 当前 stream
和 FP32 张量内存。

## Prespecified comparison / 对照协议

- Original 1.50 MB B, original t00 hyperparameters, seed43, first session
  `indy_20160622_01`, fold1, frozen preprocessing, batch128 including short tail.
- Five epochs per arm, three timing repeats of the same seed, alternating
  baseline/custom order. Keep the original cosine scheduler T_max=60.
- Three forward/backward warm-up batches, then restore weights and RNG before
  constructing AdamW/EMA. Compilation and warm-up time are separate.
- Both arms retain raw+EMA validation and fixed training probes every epoch,
  plus the original best-state CPU copies and per-batch scalar synchronization.
- Measure synchronized training time and full epoch compute+logging wall time;
  report repeat-mean sample SD, paired speedups and peak allocated/reserved VRAM.
- GPU gates: output and projected-input gradients across shapes and saturated
  inputs; full-model train/eval outputs, parameter gradients and AdamW update.
- CPU algebra/finite differences and offline compilation do NOT verify CUDA
  execution. GPU gates must pass before any timing fit.
- Tentative adoption requires >=10% mean full-epoch time reduction, positive
  speedup in each repeat, and <=0.001 raw/EMA validation R2 difference across
  matched epochs. This is a smoke-test threshold, not final R2 equivalence.
- No test data is evaluated. This one session/fold is not a general speed claim.

固定原 B/t00、seed43、第一 session 的第1折，batch128，5 epoch，重复计时
3次并交替顺序。余弦调度周期仍为60。预热后恢复模型与随机状态，编译/预热单列。
完整保留 raw/EMA 验证、训练 probe、最佳权重复制和 batch 数值同步。只有 GPU
输出、梯度和更新检查通过后才计时。CPU检查或编译成功不能当成CUDA验证成功。
平均完整 epoch 至少节省10%，每次重复均更快且各 epoch R² 差不超过0.001，
才列为候选；这不能证明完整训练后的精度等价。本实验不评估 test。

## Commands / 命令

Run from the repository root / 从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m indy_loco.experiment.phase18_large_mingru.cuda_equivalence.benchmark --cpu-check
.\.venv\Scripts\python.exe -X utf8 -m indy_loco.experiment.phase18_large_mingru.cuda_equivalence.benchmark --compile-only
.\.venv\Scripts\python.exe -X utf8 -u -m indy_loco.experiment.phase18_large_mingru.cuda_equivalence.benchmark
```

The full run shares the Phase17/18 OS GPU lock and refuses to overwrite a
completed report. Active results are under `../results/cuda_equivalence_v3`;
v1/v2 are retained failed correctness attempts, not training results. If a
partial timed run is interrupted, inspect it rather than mixing its timing with
a new run. Prior tuning does not resume automatically from this benchmark.

完整实验使用共享 GPU 系统锁，拒绝覆盖完成报告。结果写入
`../results/cuda_equivalence_v3`，v1/v2仅为未通过正确性检查的记录。
计时中断后先检查，不混合不同运行的计时。
该脚本不会自动续跑此前尚未完成的超参数调参。

References: [NVRTC](https://docs.nvidia.com/cuda/nvrtc/),
[PyTorch autograd extension](https://docs.pytorch.org/docs/stable/notes/extending.html).

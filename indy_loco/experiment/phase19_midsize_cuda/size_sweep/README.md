# Midsize paired width sweep / 配对宽度扫描

Three models: encoder/GRU widths 96/96, 128/128, 256/256. Not a 3x3 grid.
Four TCN blocks, kernel3, dilations1/2/4/8, one GRU layer, two outputs, 50-bin
windows and original dropout remain unchanged. Native PyTorch CUDA FP32 only;
Phase19's 64-channel custom kernels are not used at these new widths.

三组配对尺寸，不是九组组合。保留原架构和预处理，仅调整 encoder/decoder 宽度。

All six original sessions, all five folds, seed43: 90 jobs. Same-fold canonical
Midsize overlap transfer initializes each candidate using the Phase16 gate-aware
transfer implementation; added weights retain seeded random initialization.
Widening is not a function-preserving transform. The reference checkpoints were
validation-selected, so these tuning scores are not unbiased final estimates.

AdamW LR3e-4 for GRU/head, encoder LR7.5e-5, weight decay0.025, clip1,
batch128 including the tail, cosine schedule20 epochs. Reuse Phase16 early stop:
minimum4 epochs, patience3, relative improvement0.5%, maximum20. Select the exact
minimum validation MSE checkpoint; report its validation R2. No test inference.
Rank widths only after all 30 matched folds per width are complete. Original
Midsize and minGRU checkpoints and earlier experiments are never modified.

每组30折，三组90项；最多20 epoch，沿用现有早停和同折权重迁移。
仅根据 validation 选择，暂不评估 test，也不覆盖现有最佳模型。

```powershell
cd "C:\Users\fangz\Documents\Summer-Hand-intent-decoder-proj"
.\.venv\Scripts\python.exe -X utf8 -u -m indy_loco.experiment.phase19_midsize_cuda.size_sweep.run --resume
```

`--resume` verifies configuration/source/checkpoint/data evidence and skips
completed folds. Interrupted folds restart from their original initialization;
this is not an optimizer-state mid-epoch resume. Orphaned saved checkpoints are
preserved with an interrupted timestamp suffix. A shared Phase17/18 GPU lock
prevents concurrent cooperating training. No recurring monitor is created.

Results: `../results/size_sweep_v1/{status.json,metrics.json,config.json}`,
`fold_results/*.json`, `checkpoints/*.pt`. Each fold result contains the complete
epoch history, validation scores, timing, VRAM, hashes and preprocessing evidence.
Console logs provide live epoch progress before each fold finishes.

参数/FP32权重（十进制 MB，非运行显存）：

| Encoder / decoder | Parameters | FP32 MB |
|---|---:|---:|
| 96 / 96 | 185,762 | 0.743048 |
| 128 / 128 | 321,410 | 1.285640 |
| 256 / 256 | 1,232,642 | 4.930568 |

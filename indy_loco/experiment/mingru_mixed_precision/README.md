# minGRU INT8/FP32 quantization / minGRU 混合量化

## Baseline resolved: b2 / 已确认 b2

2026-09-30: User clarified that0.76414 was b2's earlier27-fold validation score
and authorized full b2 test evaluation, followed by quantization if it beats B0.
All30 b2 EMA checkpoints passed hash/receipt/split verification; no retraining was
needed. Full validation R²=0.7629117767; test R²=0.7586095989 ±0.0698201112 sample
SD. Paired improvement over B0=+0.001883326, with23/30 wins. Quantization started
on this b2 package using CPU2 threads. Results: `results/b2_mixed_precision_v1`.
Verified source package: `results/b2_completion_v1/package`.

The following describes the initial source discrepancy, now resolved:

User requested the 1.50 MB model with test R² about **0.76414**. The local
published `indy_loco/models/mingru_b/manifest.json` instead reports
**0.7567262729008992**, 374,402 parameters, 1,497,608 FP32 weight bytes.
Local b_tuning_v1 progress records a CUDA failure at77/78 fits and no final test
report. Do not silently substitute the published checkpoint for the requested
version. No trained-model quantization evaluation has been started at authoring.
The synthetic tests and checkpoint-independent capacity inventory have run.

0.76414是先前27折的validation。现已核实b2全部30折并完成test，优于原B，
按用户条件启动b2量化。旧发布包和原始训练检查点保留不变。

## Prespecified experiment / 预设实验

Four groups: encoder (stem + causal frontend), minGRU projections, FFNs, output
head. Enumerate all15 nonempty subsets in each of two modes, plus one FP32
baseline: **31 plans ×30 matched folds =930 fold-plan evaluations**. Evaluate
validation and test for every plan; compare paired test R² to the reproduced FP32
baseline of the SAME fold. Report mean/sample SD, mean paired change and worst
fold change. Preserve all checkpoints and frozen preprocessing; no training.

W8A32: INT8 weights, FP32 inputs/outputs/compute after dequantization.
W8A8-reference: INT8 weights and operator inputs, exact integer accumulation
simulated using FP32, followed by FP32 rescaling and bias. For this model,
K*127²<2²⁴, so integer accumulation fits the exact integer range of FP32.
Neither mode is a native integer MCU implementation or a valid speed benchmark.
Normalization, biases, GELU, gates, recurrent state, residuals and output tensors
remain FP32. There is no FP16. All weights use per-output-channel symmetric
INT8[-127,127], round-to-nearest-even, FP32 scales and fixed zero point0.

Static operator-input ranges come from1024 evenly spaced TRAIN windows in each
fold, with absmax and no percentile search. No labels, validation/test inputs or
test metrics fit these scales. Test results are exposed for the user's requested
quantization tuning; they are not an untouched final accuracy estimate.

编码器、minGRU投影、FFN、head四个分区穷举，31方案×30折。INT8只用于选中
算子的权重（W8A32）或权重和输入（W8A8参考模式），其余FP32。量化范围仅从
训练窗口确定。按用户要求比较test，但这些结果应称量化调参结果。

Full evaluation uses the existing parallel recurrence. An eight-window
validation probe compares each plan's parallel and sequential outputs; activation
rounding can amplify arithmetic differences. This probe is not a full-dataset
sequential deployment validation. Before deployment, repeat full inference with
the chosen backend's rounding, integer bias, requantization and overflow rules.

## Memory and latency boundaries / 内存与延迟

`PLAN.json` lists exact tensor payloads, FP32 scales, INT8 parameter counts and
32-byte-aligned weight totals. It does not count Python/PyTorch serialization as
deployable weight size. Universal INT8 bundles are saved per fold; a plan manifest
selects tensors from the bundle or the immutable original FP32 checkpoint.

Placement scenarios use128/256/384/512 KiB of hypothetical NET on-chip weight
space. They keep head tensors external to demonstrate SDRAM and prioritize
temporally reused tensors on chip. A partially resident tensor requires tiling.
These are not a linker layout and exclude activation/workspace, DMA staging,
stacks, peripheral buffers and existing firmware. The local STM32H747 linker
defines128 KiB DTCM and512 KiB D1 AXI SRAM; D2 is shared/occupied, not automatically
available. Actual net free capacity must be measured during integration.

Report external weight bytes for once-per-window fetching and naive per-timestep
rereading; these are traffic scenarios, not measured bandwidth/latency. Neither
smaller weights nor lower host reference time proves <40ms MCU inference. Real
mixed-precision kernels, external-SDRAM placement and board timing remain separate.

内存报告包含scale和对齐；片上预算是扣除其他用途后的假设净权重空间，不是
可用SRAM测量。head保留SDRAM。外存字节数只能支持访存分析，不能保证40ms。

## Commands / 命令

Synthetic correctness checks, no dataset/checkpoint access:

```powershell
cd "C:\Users\fangz\Documents\Summer-Hand-intent-decoder-proj"
.\.venv\Scripts\python.exe -X utf8 -m unittest indy_loco.experiment.mingru_mixed_precision.test_quantization -v
.\.venv\Scripts\python.exe -X utf8 -m indy_loco.experiment.mingru_mixed_precision.describe
```

ONLY if the user confirms using the published0.756726 package:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m indy_loco.experiment.mingru_mixed_precision.run --package-root indy_loco/models/mingru_b --expected-baseline-r2 0.7567262729008992 --device cpu --threads 2 --resume
```

For the requested0.76414 version, supply its verified package root and matching
expected R². The loader requires30 fold records, checkpoint hashes, strict ModelB
state shapes, EMA identity, preprocessing fingerprints and reference fold test
scores. If its format differs, adapt the loader to its actual metadata; do not
fabricate or edit an old manifest. Evaluation aborts on wrong baseline or failed
FP32 reproduction. CPU mode avoids taking the GPU from another training chat.
CUDA mode acquires the shared Phase17/18 GPU lock. No automatic scheduling.

Results reside in `results/mixed_precision_v1`: status/config, calibration,
int8_weights, fold_results, metrics.json and REPORT.md. Resume only accepts the
same source/config/checkpoints and skips verified completed fold-plan receipts.

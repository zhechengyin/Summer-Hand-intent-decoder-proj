# Phase19 Midsize CUDA comparison / 对比结果

Same FP32 model and data; failed numerical candidates are excluded from timing.
相同 FP32 模型和数据；未通过数值检查的候选不进入计时。

| Variant | Full epoch seconds | Training seconds | Speedup | Max validation R² delta | Candidate |
|---|---:|---:|---:|---:|---|
| pytorch | 8.5147 | 8.0896 | 1.000x | 0.000000 | False |
| epilogue | 9.3402 | 8.8666 | 0.912x | 0.000000 | False |
| causal | 9.2100 | 8.7634 | 0.925x | 0.000054 | False |
| layernorm | 7.7244 | 7.3251 | 1.102x | 0.000026 | False |

## Rejected numerical candidates / 数值检查未通过

- gru: real_batch/AdamW/gru.weight_ih_l0: {'name': 'real_batch/AdamW/gru.weight_ih_l0', 'kind': 'update', 'shape': [192, 64], 'max_absolute_error': 4.7283247113227844e-05, 'max_tolerance_ratio': 18.32797622680664, 'passed': False}
- combined: real_batch/AdamW/gru.weight_ih_l0: {'name': 'real_batch/AdamW/gru.weight_ih_l0', 'kind': 'update', 'shape': [192, 64], 'max_absolute_error': 4.783831536769867e-05, 'max_tolerance_ratio': 18.543132781982422, 'passed': False}

One session/fold/seed, warm-started from canonical fold1. Repeats measure timing, not independent accuracy seeds. Short-run validation does not establish final test R2 or STM32 speed. cuDNN may beat custom GRU. No automatic adoption.

单 session、第1折、单 seed 的短训练计时；复用该折 Midsize 初始权重。
不评估 test，不代表最终 R² 或 STM32 性能，不自动替换原模型。

#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
STEDGEAI=${STEDGEAI:-/Users/yinzhecheng/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/macarm/stedgeai}
ENCODER_MODEL="$DEPLOY_DIR/model/indy_phase6_encoder.onnx"
GRU_HEAD_MODEL="$DEPLOY_DIR/model/indy_phase6_gru_head.h5"
WORK_DIR="$DEPLOY_DIR/cubeai/workspace"
LOG_DIR="$DEPLOY_DIR/logs"

ENCODER_ROOT="$DEPLOY_DIR/cubeai/encoder"
GRU_ROOT="$DEPLOY_DIR/cubeai/gru_head"
CHAIN_ROOT="$DEPLOY_DIR/cubeai/end_to_end"
mkdir -p \
  "$ENCODER_ROOT/analyze" "$ENCODER_ROOT/validate" "$ENCODER_ROOT/generated" \
  "$GRU_ROOT/analyze" "$GRU_ROOT/validate" "$GRU_ROOT/generated" \
  "$CHAIN_ROOT" "$WORK_DIR" "$LOG_DIR" "$DEPLOY_DIR/metadata"

"$STEDGEAI" analyze \
  --target stm32 --model "$ENCODER_MODEL" --type onnx --name indy_encoder \
  --optimization time --c-api legacy \
  --workspace "$WORK_DIR/analyze_encoder" --output "$ENCODER_ROOT/analyze" \
  --verbosity 1 2>&1 | tee "$LOG_DIR/stedgeai_analyze_encoder.log"

"$STEDGEAI" validate \
  --target stm32 --model "$ENCODER_MODEL" --type onnx --name indy_encoder \
  --optimization time --c-api legacy --mode host \
  --valinput "$DEPLOY_DIR/validation/validation_inputs.csv" \
  --valoutput "$DEPLOY_DIR/validation/encoder_pytorch_outputs.csv" \
  --workspace "$WORK_DIR/validate_encoder" --output "$ENCODER_ROOT/validate" \
  --save-csv --verbosity 1 2>&1 | tee "$LOG_DIR/stedgeai_validate_encoder.log"

"$STEDGEAI" generate \
  --target stm32 --model "$ENCODER_MODEL" --type onnx --name indy_encoder \
  --optimization time --c-api legacy \
  --workspace "$WORK_DIR/generate_encoder" --output "$ENCODER_ROOT/generated" \
  --verbosity 1 2>&1 | tee "$LOG_DIR/stedgeai_generate_encoder.log"

"$STEDGEAI" analyze \
  --target stm32 --model "$GRU_HEAD_MODEL" --type keras --name indy_gru_head \
  --optimization time --c-api legacy \
  --workspace "$WORK_DIR/analyze_gru_head" --output "$GRU_ROOT/analyze" \
  --verbosity 1 2>&1 | tee "$LOG_DIR/stedgeai_analyze_gru_head.log"

"$STEDGEAI" validate \
  --target stm32 --model "$GRU_HEAD_MODEL" --type keras --name indy_gru_head \
  --optimization time --c-api legacy --mode host \
  --valinput "$DEPLOY_DIR/validation/encoder_pytorch_outputs.csv" \
  --valoutput "$DEPLOY_DIR/validation/pytorch_outputs.csv" \
  --workspace "$WORK_DIR/validate_gru_head" --output "$GRU_ROOT/validate" \
  --save-csv --verbosity 1 2>&1 | tee "$LOG_DIR/stedgeai_validate_gru_head.log"

"$STEDGEAI" generate \
  --target stm32 --model "$GRU_HEAD_MODEL" --type keras --name indy_gru_head \
  --optimization time --c-api legacy \
  --workspace "$WORK_DIR/generate_gru_head" --output "$GRU_ROOT/generated" \
  --verbosity 1 2>&1 | tee "$LOG_DIR/stedgeai_generate_gru_head.log"

# Validate the generated-C chain by feeding encoder C outputs into GRU-head C.
"$STEDGEAI" validate \
  --target stm32 --model "$GRU_HEAD_MODEL" --type keras --name indy_gru_head \
  --optimization time --c-api legacy --mode host \
  --valinput "$ENCODER_ROOT/validate/indy_encoder_val_c_outputs_1.csv" \
  --valoutput "$DEPLOY_DIR/validation/pytorch_outputs.csv" \
  --workspace "$WORK_DIR/validate_end_to_end" --output "$CHAIN_ROOT" \
  --save-csv --verbosity 1 2>&1 | tee "$LOG_DIR/stedgeai_validate_end_to_end.log"

"$STEDGEAI" --version > "$DEPLOY_DIR/metadata/stedgeai_version.txt"
echo "Cube.AI split-model artifacts written below $DEPLOY_DIR/cubeai"

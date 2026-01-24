#!/usr/bin/env bash
set -euo pipefail

# Real training for Heads on all datasets with GPU
# Prereqs: export DASHSCOPE_API_KEY (optional DASHSCOPE_BASE_URL), ensure encoder under model/encoder and GPU available.

DATASETS=(
  "datasets/gsm8k_arrow"
  "datasets/hendrycks_math_arrow"
)

RUN_DIR="runs/heads_ckpt"
DEVICE="cuda"
SPLIT="train"

: "${DASHSCOPE_API_KEY:?Please export DASHSCOPE_API_KEY before running this script.}"

for ds in "${DATASETS[@]}"; do
  echo "=== Training on $ds ==="
  python src/run.py \
    --dataset-path "$ds" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --save-dir "$RUN_DIR"
done

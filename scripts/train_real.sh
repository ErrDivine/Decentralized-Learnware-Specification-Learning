#!/usr/bin/env bash
set -euo pipefail

# Real training for Heads on all datasets with GPU
# Prereqs: export DASHSCOPE_API_KEY, ensure models under model/ and GPU available.

DATASETS=(
  "datasets/gsm8k_arrow"
  "datasets/hendrycks_math_arrow"
)

RUN_DIR="runs/heads_ckpt"
DEVICE="cuda"
SPLIT="train"

export DASHSCOPE_API_KEY='sk-46f61c60859f4d19a1de714803d10f3e'

for ds in "${DATASETS[@]}"; do
  echo "=== Training on $ds ==="
  python src/run.py \
    --dataset-path "$ds" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --save-dir "$RUN_DIR"
done

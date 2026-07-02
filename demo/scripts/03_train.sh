#!/usr/bin/env bash
# 03_train.sh - 启动 LLaMA-Factory 单 H800 全量 SFT
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT=demo
CONFIG=$ROOT/config/demo.yaml
OUT_DIR=$ROOT/output/onereason_0.8b_sft
LOG=$OUT_DIR/train.log

source $ROOT/LLaMA-Factory/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=1

mkdir -p $OUT_DIR

echo "[run] llamafactory-cli train $CONFIG"
echo "[run] log -> $LOG"
llamafactory-cli train $CONFIG 2>&1 | tee $LOG

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU_ID=${GPU_ID:-3}
CONFIG=${CONFIG:-demo/config/server_a100.yaml}
MIN_FREE_MIB=${MIN_FREE_MIB:-60000}
LOG=${LOG:-demo/output/train_server_$(date +%Y%m%d_%H%M%S).log}
VENV=demo/LLaMA-Factory/.venv

if [[ ! -x "$VENV/bin/llamafactory-cli" ]]; then
  echo "[error] run bash demo/scripts/setup_server.sh first" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[error] config not found: $CONFIG" >&2
  exit 1
fi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
if (( FREE_MIB < MIN_FREE_MIB )); then
  echo "[error] GPU $GPU_ID has ${FREE_MIB} MiB free; require at least ${MIN_FREE_MIB} MiB" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=1

mkdir -p demo/output
echo "[run] GPU=$GPU_ID free=${FREE_MIB}MiB config=$CONFIG"
echo "[run] log=$LOG"
"$VENV/bin/llamafactory-cli" train "$CONFIG" 2>&1 | tee "$LOG"

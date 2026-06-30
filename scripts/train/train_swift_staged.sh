#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
STAGE1_MAX_LENGTH="${STAGE1_MAX_LENGTH:-2048}"
MIN_FREE_MIB="${MIN_FREE_MIB:-10000}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-2}"
STAGE1_GRAD_ACCUM="${STAGE1_GRAD_ACCUM:-4}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-1}"
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE2_LR="${STAGE2_LR:-5e-5}"
STAGE1_OUTPUT="${STAGE1_OUTPUT:-${ROOT}/outputs/swift_lora_stage1_material}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-${ROOT}/outputs/swift_lora_stage2_user_rec}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"

IFS=',' read -r -a GPU_IDS <<<"${CUDA_VISIBLE_DEVICES}"
for gpu in "${GPU_IDS[@]}"; do
  if [[ "${gpu}" == "2" || "${gpu}" == "3" ]]; then
    echo "Refusing to use protected GPU ${gpu}." >&2
    exit 2
  fi
done
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE="${#GPU_IDS[@]}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

check_memory() {
  for gpu in "${GPU_IDS[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    echo "GPU ${gpu}: ${free_mib} MiB free"
    if (( free_mib < MIN_FREE_MIB )); then
      echo "GPU ${gpu} is below the ${MIN_FREE_MIB} MiB safety threshold; aborting before allocation." >&2
      exit 3
    fi
  done
}

cd "${ROOT}"
python scripts/data/build_staged_sft.py

if [[ "${RUN_STAGE1}" == "1" ]]; then
  check_memory
  TRAIN_DATA="${ROOT}/data/processed/stage1_material_train.jsonl" \
  VALID_DATA="${ROOT}/data/processed/stage1_material_valid.jsonl" \
  OUTPUT_DIR="${STAGE1_OUTPUT}" MAX_LENGTH="${STAGE1_MAX_LENGTH}" LR="${STAGE1_LR}" \
  BATCH_SIZE="${STAGE1_BATCH_SIZE}" GRAD_ACCUM="${STAGE1_GRAD_ACCUM}" \
  SAVE_STEPS=500 EVAL_STEPS=500 DATASET_NUM_PROC="${DATASET_NUM_PROC}" \
  bash scripts/train/train_swift_lora.sh
fi

stage1_adapter="$(find "${STAGE1_OUTPUT}" -type f -name adapter_model.safetensors -printf '%T@ %h\n' \
  | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "${stage1_adapter}" ]]; then
  echo "No stage-1 adapter checkpoint found under ${STAGE1_OUTPUT}." >&2
  exit 4
fi
echo "Stage-2 initialization adapter: ${stage1_adapter}"

if [[ "${RUN_STAGE2}" == "1" ]]; then
  check_memory
  TRAIN_DATA="${ROOT}/data/processed/stage2_user_rec_train.jsonl" \
  VALID_DATA="${ROOT}/data/processed/stage2_user_rec_valid.jsonl" \
  OUTPUT_DIR="${STAGE2_OUTPUT}" MAX_LENGTH="${MAX_LENGTH}" LR="${STAGE2_LR}" \
  BATCH_SIZE="${STAGE2_BATCH_SIZE}" GRAD_ACCUM="${STAGE2_GRAD_ACCUM}" \
  SAVE_STEPS=500 EVAL_STEPS=500 DATASET_NUM_PROC="${DATASET_NUM_PROC}" \
  ADAPTERS="${stage1_adapter}" bash scripts/train/train_swift_lora.sh
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
MODEL_PATH="${MODEL_PATH:-/data/hz/models/OneReason-0.8B-pretrain-competition}"
DATA_ROOT="${DATA_ROOT:-/data/hz/onereason_competition/data/material_2ep/swift_messages_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/material_2ep_allfull_gpu0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EP1_LR="${EP1_LR:-1e-5}"
EP2_LR="${EP2_LR:-5e-6}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
RUN_STAGE="${RUN_STAGE:-both}"
EP1_RESUME="${EP1_RESUME:-}"
EP2_RESUME="${EP2_RESUME:-}"

if [[ "${CUDA_VISIBLE_DEVICES}" != "0" ]]; then
  echo "This run is restricted to physical GPU 0; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi
if [[ "${RUN_STAGE}" != "both" && "${RUN_STAGE}" != "ep1" && "${RUN_STAGE}" != "ep2" ]]; then
  echo "RUN_STAGE must be both, ep1, or ep2." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE=1
export MASTER_PORT="${MASTER_PORT:-29680}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

for file in ep1_train.jsonl ep1_valid.jsonl ep2_train.jsonl ep2_valid.jsonl; do
  test -s "${DATA_ROOT}/${file}" || { echo "Missing dataset: ${DATA_ROOT}/${file}" >&2; exit 3; }
done

latest_checkpoint() {
  find "$1" -type f -name model.safetensors -printf '%T@ %h\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_full() {
  local model="$1" train="$2" valid="$3" output="$4" lr="$5" resume="$6"
  MODEL_PATH="${model}" TRAIN_DATA="${train}" VALID_DATA="${valid}" OUTPUT_DIR="${output}" \
  MAX_LENGTH="${MAX_LENGTH}" LR="${lr}" EPOCHS=1 BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM="${GRAD_ACCUM}" SAVE_STEPS="${SAVE_STEPS}" EVAL_STEPS="${EVAL_STEPS}" \
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" DATASET_NUM_PROC="${DATASET_NUM_PROC}" \
  RESUME_FROM_CHECKPOINT="${resume}" REPORT_TO=tensorboard \
  bash "${ROOT}/scripts/train/train_swift_full.sh"
}

mkdir -p "${OUTPUT_ROOT}"
if [[ "${RUN_STAGE}" == "both" || "${RUN_STAGE}" == "ep1" ]]; then
  run_full "${MODEL_PATH}" "${DATA_ROOT}/ep1_train.jsonl" "${DATA_ROOT}/ep1_valid.jsonl" \
    "${OUTPUT_ROOT}/ep1_think" "${EP1_LR}" "${EP1_RESUME}"
fi

EP1_MODEL="${EP1_MODEL:-$(latest_checkpoint "${OUTPUT_ROOT}/ep1_think")}" 
if [[ "${RUN_STAGE}" == "both" || "${RUN_STAGE}" == "ep2" ]]; then
  test -n "${EP1_MODEL}" || { echo "No EP1 full checkpoint found." >&2; exit 4; }
  echo "EP2 initialization model: ${EP1_MODEL}"
  run_full "${EP1_MODEL}" "${DATA_ROOT}/ep2_train.jsonl" "${DATA_ROOT}/ep2_valid.jsonl" \
    "${OUTPUT_ROOT}/ep2_no_think" "${EP2_LR}" "${EP2_RESUME}"
fi

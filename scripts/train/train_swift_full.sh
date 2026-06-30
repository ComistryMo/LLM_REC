#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
MODEL_PATH="${MODEL_PATH:-/data/hz/models/OneReason-0.8B-pretrain-competition}"
MODEL_TYPE="${MODEL_TYPE:-qwen3}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-qwen3_thinking}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data/processed/train_official_v1.jsonl}"
VALID_DATA="${VALID_DATA:-${ROOT}/data/processed/valid_official_v1.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/swift_full_v1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
DEEPSPEED="${DEEPSPEED:-${ROOT}/configs/deepspeed/zero2.json}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
REPORT_TO="${REPORT_TO:-tensorboard}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "${OUTPUT_DIR}" "${ROOT}/logs"
cd "${ROOT}"
python scripts/data/check_format.py "${TRAIN_DATA#${ROOT}/}" "${VALID_DATA#${ROOT}/}" --max-length "${MAX_LENGTH}"

ARGS=(
  sft
  --model "${MODEL_PATH}"
  --model_type "${MODEL_TYPE}"
  --template "${TEMPLATE_TYPE}"
  --dataset "${TRAIN_DATA}"
  --val_dataset "${VALID_DATA}"
  --output_dir "${OUTPUT_DIR}"
  --tuner_type full
  --torch_dtype bfloat16
  --bf16 true
  --attn_impl "${ATTN_IMPL}"
  --max_length "${MAX_LENGTH}"
  --learning_rate "${LR}"
  --num_train_epochs "${EPOCHS}"
  --per_device_train_batch_size "${BATCH_SIZE}"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --gradient_checkpointing true
  --loss_scale default
  --deepspeed "${DEEPSPEED}"
  --eval_strategy steps
  --eval_steps "${EVAL_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit 2
  --logging_steps 5
  --report_to "${REPORT_TO}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --dataloader_num_workers 0
)
if [[ -n "${MAX_STEPS}" ]]; then ARGS+=(--max_steps "${MAX_STEPS}"); fi

printf '%q ' swift "${ARGS[@]}" | tee "${ROOT}/logs/train_swift_full_cmd.log"
echo
swift "${ARGS[@]}" 2>&1 | tee "${ROOT}/logs/train_swift_full.log"

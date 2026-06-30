#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
MODEL_PATH="${MODEL_PATH:-/data/hz/models/OneReason-0.8B-pretrain-competition}"
MODEL_TYPE="${MODEL_TYPE:-qwen3}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-qwen3_thinking}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data/processed/train_official_v1.jsonl}"
VALID_DATA="${VALID_DATA:-${ROOT}/data/processed/valid_official_v1.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/swift_lora_v1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
REPORT_TO="${REPORT_TO:-tensorboard}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
ADAPTERS="${ADAPTERS:-}"
QUANT_BITS="${QUANT_BITS:-}"
QUANT_METHOD="${QUANT_METHOD:-bnb}"

mkdir -p "${OUTPUT_DIR}" "${ROOT}/logs"
cd "${ROOT}"
python scripts/data/check_format.py "${TRAIN_DATA#${ROOT}/}" "${VALID_DATA#${ROOT}/}" --max-length "${MAX_LENGTH}"
read -r -a TARGET_MODULE_LIST <<<"${TARGET_MODULES}"

ARGS=(
  sft
  --model "${MODEL_PATH}"
  --model_type "${MODEL_TYPE}"
  --template "${TEMPLATE_TYPE}"
  --dataset "${TRAIN_DATA}"
  --val_dataset "${VALID_DATA}"
  --output_dir "${OUTPUT_DIR}"
  --tuner_type lora
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
  --ddp_find_unused_parameters false
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --target_modules "${TARGET_MODULE_LIST[@]}"
  --loss_scale default
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
if [[ -n "${ADAPTERS}" ]]; then ARGS+=(--adapters "${ADAPTERS}"); fi
if [[ -n "${QUANT_BITS}" ]]; then
  ARGS+=(
    --quant_bits "${QUANT_BITS}"
    --quant_method "${QUANT_METHOD}"
    --bnb_4bit_compute_dtype bfloat16
    --bnb_4bit_quant_type nf4
    --bnb_4bit_use_double_quant true
  )
fi

printf '%q ' swift "${ARGS[@]}" | tee "${ROOT}/logs/train_swift_lora_cmd.log"
echo
swift "${ARGS[@]}" 2>&1 | tee "${ROOT}/logs/train_swift_lora.log"

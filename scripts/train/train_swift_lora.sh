#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
MODEL_PATH="${MODEL_PATH:-/data/hz/models/OneReason-0.8B-pretrain-competition}"
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

mkdir -p "${OUTPUT_DIR}" "${ROOT}/logs"
cd "${ROOT}"
python scripts/data/check_format.py "${TRAIN_DATA#${ROOT}/}" "${VALID_DATA#${ROOT}/}" --max-length "${MAX_LENGTH}"

HELP="$(swift sft --help 2>&1 || true)"
has_arg() { grep -q -- "$1" <<<"${HELP}"; }

ARGS=(sft --model "${MODEL_PATH}" --dataset "${TRAIN_DATA}" --output_dir "${OUTPUT_DIR}")
if has_arg "--val_dataset"; then ARGS+=(--val_dataset "${VALID_DATA}"); fi
if has_arg "--train_type"; then ARGS+=(--train_type lora); elif has_arg "--tuner_type"; then ARGS+=(--tuner_type lora); fi
if has_arg "--torch_dtype"; then ARGS+=(--torch_dtype bfloat16); elif has_arg "--dtype"; then ARGS+=(--dtype bf16); fi
if has_arg "--max_length"; then ARGS+=(--max_length "${MAX_LENGTH}"); fi
if has_arg "--learning_rate"; then ARGS+=(--learning_rate "${LR}"); fi
if has_arg "--num_train_epochs"; then ARGS+=(--num_train_epochs "${EPOCHS}"); fi
if has_arg "--per_device_train_batch_size"; then ARGS+=(--per_device_train_batch_size "${BATCH_SIZE}"); fi
if has_arg "--gradient_accumulation_steps"; then ARGS+=(--gradient_accumulation_steps "${GRAD_ACCUM}"); fi
if has_arg "--gradient_checkpointing"; then ARGS+=(--gradient_checkpointing true); fi
if has_arg "--lora_rank"; then ARGS+=(--lora_rank "${LORA_RANK}"); fi
if has_arg "--lora_alpha"; then ARGS+=(--lora_alpha "${LORA_ALPHA}"); fi
if has_arg "--lora_dropout"; then ARGS+=(--lora_dropout "${LORA_DROPOUT}"); fi
if has_arg "--target_modules"; then ARGS+=(--target_modules all-linear); fi
if has_arg "--assistant_only_loss"; then ARGS+=(--assistant_only_loss true); fi

printf 'swift %q ' "${ARGS[@]}" | tee "${ROOT}/logs/train_swift_lora_cmd.log"
echo
swift "${ARGS[@]}" 2>&1 | tee "${ROOT}/logs/train_swift_lora.log"

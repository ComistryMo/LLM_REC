#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
MODEL_PATH="${MODEL_PATH:-/data/hz/models/OneReason-0.8B-pretrain-competition}"
ADAPTER_DIR="${ADAPTER_DIR:-${ROOT}/outputs/llamafactory_lora_v1}"
EXPORT_DIR="${EXPORT_DIR:-${ROOT}/outputs/export_llamafactory_lora_v1}"

mkdir -p "${EXPORT_DIR}" "${ROOT}/logs"
llamafactory-cli export \
  --model_name_or_path "${MODEL_PATH}" \
  --adapter_name_or_path "${ADAPTER_DIR}" \
  --template default \
  --finetuning_type lora \
  --export_dir "${EXPORT_DIR}" \
  --export_size 5 \
  --export_device cpu \
  --export_legacy_format false 2>&1 | tee "${ROOT}/logs/export_llamafactory_lora.log"

python "${ROOT}/scripts/export/pre_submit_check.py" --model-dir "${EXPORT_DIR}"


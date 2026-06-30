#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
MODEL_PATH="${MODEL_PATH:-/data/hz/models/OneReason-0.8B-pretrain-competition}"
ADAPTER_DIR="${ADAPTER_DIR:-${ROOT}/outputs/swift_lora_v1}"
EXPORT_DIR="${EXPORT_DIR:-${ROOT}/outputs/export_swift_lora_v1}"

mkdir -p "${EXPORT_DIR}" "${ROOT}/logs"
HELP="$(swift export --help 2>&1 || true)"
has_arg() { grep -q -- "$1" <<<"${HELP}"; }

ARGS=(export --model "${MODEL_PATH}" --adapters "${ADAPTER_DIR}" --output_dir "${EXPORT_DIR}")
if has_arg "--merge_lora"; then ARGS+=(--merge_lora true); fi
if has_arg "--safe_serialization"; then ARGS+=(--safe_serialization true); fi

swift "${ARGS[@]}" 2>&1 | tee "${ROOT}/logs/export_swift_lora.log"
python "${ROOT}/scripts/export/pre_submit_check.py" --model-dir "${EXPORT_DIR}"


#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
MODEL_PATH="${MODEL_PATH:-/data/hz/models/OneReason-0.8B-pretrain-competition}"
ADAPTER_DIR="${ADAPTER_DIR:-${ROOT}/outputs/swift_lora_v1}"
EXPORT_DIR="${EXPORT_DIR:-${ROOT}/outputs/export_swift_lora_v1}"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-${MODEL_PATH}/config.json}"

mkdir -p "$(dirname "${EXPORT_DIR}")" "${ROOT}/logs"
if [[ -e "${EXPORT_DIR}" ]]; then
  echo "EXPORT_DIR already exists: ${EXPORT_DIR}" >&2
  exit 2
fi
HELP="$(swift export --help 2>&1 || true)"
has_arg() { grep -q -- "$1" <<<"${HELP}"; }

ARGS=(export --model "${MODEL_PATH}" --adapters "${ADAPTER_DIR}" --output_dir "${EXPORT_DIR}")
if has_arg "--merge_lora"; then ARGS+=(--merge_lora true); fi
if has_arg "--safe_serialization"; then ARGS+=(--safe_serialization true); fi

swift "${ARGS[@]}" 2>&1 | tee "${ROOT}/logs/export_swift_lora.log"
if [[ ! -d "${EXPORT_DIR}" && -d "${ADAPTER_DIR}-merged" ]]; then
  # Swift 4.3.x writes QLoRA merges beside the adapter even when output_dir is set.
  cp -a "${ADAPTER_DIR}-merged" "${EXPORT_DIR}"
fi
if [[ -f "${BASE_CONFIG_PATH}" ]]; then
  # Competition evaluation validates baseline architecture fields strictly.
  cp "${BASE_CONFIG_PATH}" "${EXPORT_DIR}/config.json"
fi
python "${ROOT}/scripts/export/pre_submit_check.py" --model-dir "${EXPORT_DIR}"

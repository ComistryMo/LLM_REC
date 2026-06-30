#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
CONFIG="${CONFIG:-${ROOT}/configs/llamafactory/lora_sft.yaml}"
MODEL_PATH="${MODEL_PATH:-/data/hz/models/OneReason-0.8B-pretrain-competition}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/llamafactory_lora_v1}"

mkdir -p "${OUTPUT_DIR}" "${ROOT}/logs"
cd "${ROOT}"
python scripts/data/check_format.py data/processed/train_mix_v1.jsonl data/processed/valid_mix_v1.jsonl
cp configs/llamafactory/dataset_info.json data/processed/dataset_info.json

TMP_CONFIG="${OUTPUT_DIR}/lora_sft.resolved.yaml"
python - <<PY
from pathlib import Path
cfg = Path("${CONFIG}").read_text(encoding="utf-8")
cfg = cfg.replace("__MODEL_PATH__", "${MODEL_PATH}").replace("__OUTPUT_DIR__", "${OUTPUT_DIR}")
Path("${TMP_CONFIG}").write_text(cfg, encoding="utf-8")
PY

llamafactory-cli train "${TMP_CONFIG}" 2>&1 | tee "${ROOT}/logs/train_llamafactory_lora.log"

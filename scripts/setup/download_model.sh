#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-OpenOneRec/OneReason-0.8B-pretrain-competition}"
TARGET_DIR="${TARGET_DIR:-/data/hz/models/OneReason-0.8B-pretrain-competition}"

mkdir -p "${TARGET_DIR}"
echo "Downloading ${MODEL_ID} to ${TARGET_DIR}"
echo "Optional: export HF_ENDPOINT=https://hf-mirror.com before running if HuggingFace access is slow."

if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" --local-dir "${TARGET_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" --local-dir "${TARGET_DIR}" --local-dir-use-symlinks False
elif command -v git-lfs >/dev/null 2>&1; then
  tmp_dir="$(mktemp -d)"
  git lfs install
  git clone "https://huggingface.co/${MODEL_ID}" "${tmp_dir}/model"
  rsync -a "${tmp_dir}/model/" "${TARGET_DIR}/"
  rm -rf "${tmp_dir}"
else
  echo "Neither huggingface-cli nor git-lfs is available." >&2
  exit 1
fi

ls -lh "${TARGET_DIR}"

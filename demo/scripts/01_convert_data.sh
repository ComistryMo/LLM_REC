#!/usr/bin/env bash
# 01_convert_data.sh - 把 baseline_data 的 64 parquet 转 Alpaca JSONL
# 输出: data/data_final.jsonl + convert_summary.json + convert_filter.log
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT=demo
OUT=$ROOT/data/data_final.jsonl

source $ROOT/LLaMA-Factory/.venv/bin/activate

if [ -s "$OUT" ]; then
  echo "[skip] $OUT already exists ($(wc -l < "$OUT") lines); rerun by deleting it first"
  exit 0
fi

python $ROOT/convertv2.py \
  --input  $ROOT/baseline-data/baseline_data \
  --output $OUT \
  --summary $ROOT/data/convert_summary.json \
  --filter-log $ROOT/data/convert_filter.log \
  --max_token_types 3 \
  --shuffle --shuffle-seed 2026 \
  --report

echo "[ok] wrote $(wc -l < $OUT) records to $OUT"

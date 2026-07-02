#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

INPUT=${INPUT:?Set INPUT to a downloaded JSONL file, directory, or glob}
INPUT_FORMAT=${INPUT_FORMAT:-auto}
OUTPUT=${OUTPUT:-demo/data/competition_full.jsonl}
DATASET_NAME=${DATASET_NAME:-competition_full}
VENV=demo/LLaMA-Factory/.venv

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] run bash demo/scripts/setup_server.sh first" >&2
  exit 1
fi

if [[ "$INPUT_FORMAT" == "auto" ]]; then
  if [[ -d "$INPUT" ]] && find "$INPUT" -type f -name '*.parquet' -print -quit | grep -q .; then
    INPUT_FORMAT=parquet
  elif [[ "$INPUT" == *.parquet ]]; then
    INPUT_FORMAT=parquet
  else
    INPUT_FORMAT=jsonl
  fi
fi

case "$INPUT_FORMAT" in
  parquet)
    "$VENV/bin/python" demo/convertv2.py \
      --input "$INPUT" \
      --output "$OUTPUT" \
      --summary demo/data/convert_summary.json \
      --filter-log demo/data/convert_filter.log \
      --max_token_types 3 \
      --shuffle --shuffle-seed 2026 --report
    ;;
  jsonl)
    "$VENV/bin/python" demo/convert_jsonl.py \
      --input "$INPUT" \
      --output "$OUTPUT" \
      --shuffle --shuffle-seed 2026
    ;;
  *)
    echo "[error] INPUT_FORMAT must be auto, parquet, or jsonl" >&2
    exit 2
    ;;
esac

"$VENV/bin/python" demo/scripts/register_server_dataset.py \
  --name "$DATASET_NAME" --data "$OUTPUT"

echo "[ok] prepared $(wc -l < "$OUTPUT") records in $OUTPUT"

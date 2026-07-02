#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

INPUT=${INPUT:?Set INPUT to a downloaded JSONL file, directory, or glob}
OUTPUT=${OUTPUT:-demo/data/server_data_final.jsonl}
VENV=demo/LLaMA-Factory/.venv

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] run bash demo/scripts/setup_server.sh first" >&2
  exit 1
fi

"$VENV/bin/python" demo/convert_jsonl.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --shuffle --shuffle-seed 2026

"$VENV/bin/python" demo/scripts/register_server_dataset.py \
  --name server_data_final --data "$OUTPUT"

echo "[ok] prepared $(wc -l < "$OUTPUT") records in $OUTPUT"

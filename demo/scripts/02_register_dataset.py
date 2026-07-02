#!/usr/bin/env python3
"""Upsert data_final entry into LLaMA-Factory/data/dataset_info.json (idempotent).

NOTE: LLaMA-Factory resolves `file_name` relative to `dataset_dir`, so we write
an absolute path resolved at script run time. The script script is expected
to be run from the parent dir of `demo/` (which run_all.sh / 03_train.sh do).
"""
import json
import pathlib
import sys

INFO_PATH = pathlib.Path("demo/LLaMA-Factory/data/dataset_info.json").resolve()
DATA_PATH = str(pathlib.Path("demo/data/data_final.jsonl").resolve())

ENTRY = {
    "file_name": DATA_PATH,
    "formatting": "alpaca",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "history": "history",
    },
}

info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
info["data_final"] = ENTRY
INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] upserted 'data_final' in {INFO_PATH}", file=sys.stderr)
print(json.dumps(info["data_final"], ensure_ascii=False, indent=2))

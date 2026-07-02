#!/usr/bin/env python3
"""jsonl (list of {system, prompt, response}) -> Alpaca jsonl converter.

Each input line is expected to be a JSON list containing one dict with keys:
  - system   -> instruction
  - prompt   -> input
  - response -> output

Falls back gracefully if the line is itself a dict.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path


def iter_records(path: str):
    with open(path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] {path}:{line_no} JSON parse error: {e}", file=sys.stderr)
                continue

            if isinstance(obj, list):
                if not obj:
                    continue
                obj = obj[0]
            if not isinstance(obj, dict):
                print(f"[WARN] {path}:{line_no} unexpected type {type(obj)}", file=sys.stderr)
                continue

            system = obj.get("system", "") or ""
            prompt = obj.get("prompt", "") or ""
            response = obj.get("response", "") or ""

            if not prompt and not response:
                continue

            yield {
                "instruction": system,
                "input": prompt,
                "output": response,
                "history": [],
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="jsonl file/dir/glob (directory will be scanned for *.jsonl)")
    parser.add_argument("--output", required=True, help="output Alpaca jsonl path")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        files = sorted(str(p) for p in in_path.rglob("*.jsonl"))
    elif any(c in args.input for c in "*?["):
        files = sorted(glob.glob(args.input, recursive=True))
    else:
        files = [args.input]

    if not files:
        print(f"[ERROR] no jsonl files found under {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] found {len(files)} jsonl files", file=sys.stderr)
    for f in files:
        print(f"  - {f}", file=sys.stderr)

    records = []
    per_file_counts = {}
    for f in files:
        before = len(records)
        for rec in iter_records(f):
            records.append(rec)
        per_file_counts[f] = len(records) - before
        print(f"[INFO] {os.path.basename(f)}: {per_file_counts[f]} records", file=sys.stderr)

    print(f"[INFO] total records: {len(records)}", file=sys.stderr)

    if args.shuffle:
        random.seed(args.shuffle_seed)
        random.shuffle(records)
        print(f"[INFO] shuffled with seed {args.shuffle_seed}", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(records)} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.format_check import check_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate unified SFT JSONL format.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    ok = True
    for path in args.paths:
        report = check_records(ROOT / path, max_length=args.max_length)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        ok = ok and report["ok"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()


#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.parse_sft import parse_sft_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SFT data into unified messages JSONL.")
    parser.add_argument("inputs", nargs="+", help="Input JSONL/JSON/CSV/Parquet files.")
    parser.add_argument("--output", default="data/processed/sft_unified.jsonl")
    args = parser.parse_args()
    count = parse_sft_files(args.inputs, ROOT / args.output)
    print(f"Wrote {count} records to {ROOT / args.output}")


if __name__ == "__main__":
    main()


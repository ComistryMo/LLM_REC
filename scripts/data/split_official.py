#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.split_dataset import split_official_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe official train/valid split.")
    parser.add_argument("--input", default="data/processed/sft_unified.jsonl")
    parser.add_argument("--output-train", default="data/processed/train_official_v1.jsonl")
    parser.add_argument("--output-valid", default="data/processed/valid_official_v1.jsonl")
    parser.add_argument("--report", default="data/eda/OFFICIAL_SPLIT_REPORT.json")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()
    report = split_official_dataset(
        ROOT / args.input,
        ROOT / args.output_train,
        ROOT / args.output_valid,
        ROOT / args.report,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        max_length=args.max_length,
        deduplicate=not args.keep_duplicates,
        llamafactory_prefix=ROOT / "data/processed/llamafactory_official_v1",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

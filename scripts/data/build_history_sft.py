#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.build_history import build_history_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Build user and recommendation SFT samples from behavior sequences.")
    parser.add_argument("--input", default="data/processed/user_sequences.jsonl")
    parser.add_argument("--output-user", default="data/processed/D_user.jsonl")
    parser.add_argument("--output-rec", default="data/processed/D_rec.jsonl")
    parser.add_argument("--windows", default="20,50,100")
    parser.add_argument("--max-samples-per-user", type=int, default=12)
    args = parser.parse_args()
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    user_count, rec_count = build_history_sft(
        ROOT / args.input,
        ROOT / args.output_user,
        ROOT / args.output_rec,
        windows=windows,
        max_samples_per_user=args.max_samples_per_user,
    )
    print(f"Wrote {user_count} user samples and {rec_count} rec samples.")


if __name__ == "__main__":
    main()


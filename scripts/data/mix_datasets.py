#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.mix_datasets import mix_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix material/user/rec/world datasets by ratio.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/sft_unified.jsonl",
            "data/processed/D_user.jsonl",
            "data/processed/D_rec.jsonl",
        ],
    )
    parser.add_argument("--ratios", default="configs/mix_ratios_v1.yaml")
    parser.add_argument("--output-train", default="data/processed/train_mix_v1.jsonl")
    parser.add_argument("--output-valid", default="data/processed/valid_mix_v1.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    train_count, valid_count = mix_datasets(
        [ROOT / p for p in args.inputs],
        ROOT / args.output_train,
        ROOT / args.output_valid,
        ROOT / args.ratios if args.ratios else None,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        output_llamafactory_prefix=ROOT / "data/processed/llamafactory_mix_v1",
    )
    print(f"Wrote {train_count} train and {valid_count} valid records.")


if __name__ == "__main__":
    main()


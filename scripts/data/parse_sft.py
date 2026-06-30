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
    parser.add_argument(
        "--tokenizer",
        help="Optional local model/tokenizer path for exact token length statistics.",
    )
    args = parser.parse_args()
    length_fn = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        length_fn = lambda text: len(tokenizer.encode(text or "", add_special_tokens=False))
    kwargs = {"length_fn": length_fn} if length_fn else {}
    count = parse_sft_files(args.inputs, ROOT / args.output, **kwargs)
    print(f"Wrote {count} records to {ROOT / args.output}")


if __name__ == "__main__":
    main()

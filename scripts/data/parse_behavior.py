#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.parse_behavior import parse_behavior_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse behavior logs into chronological user sequences.")
    parser.add_argument("inputs", nargs="+", help="Input behavior JSONL/JSON/CSV/Parquet files.")
    parser.add_argument("--schema", default="configs/behavior_schema_aliases.yaml")
    parser.add_argument("--output", default="data/processed/user_sequences.jsonl")
    parser.add_argument("--events-output", default="data/interim/behavior_events.jsonl")
    args = parser.parse_args()
    schema = ROOT / args.schema if args.schema else None
    seq_count, event_count = parse_behavior_files(
        args.inputs,
        ROOT / args.output,
        ROOT / args.events_output if args.events_output else None,
        schema if schema and schema.exists() else None,
    )
    print(f"Wrote {seq_count} users and {event_count} events.")


if __name__ == "__main__":
    main()


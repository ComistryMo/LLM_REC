#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build material-first staged SFT datasets.")
    parser.add_argument("--train", default="data/processed/train_official_v1.jsonl")
    parser.add_argument("--valid", default="data/processed/valid_official_v1.jsonl")
    parser.add_argument("--user-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.user_repeat < 1:
        raise ValueError("user-repeat must be at least 1")

    train = read_jsonl(ROOT / args.train)
    valid = read_jsonl(ROOT / args.valid)
    material_train = [r for r in train if r.get("task_type") == "material_understanding"]
    material_valid = [r for r in valid if r.get("task_type") == "material_understanding"]
    rec_train = [r for r in train if r.get("task_type") == "recommendation"]
    user_train = [r for r in train if r.get("task_type") == "user_demand_understanding"]
    user_rec_valid = [
        r for r in valid if r.get("task_type") in {"user_demand_understanding", "recommendation"}
    ]
    user_rec_train = rec_train + user_train * args.user_repeat

    random.Random(args.seed).shuffle(material_train)
    random.Random(args.seed + 1).shuffle(material_valid)
    random.Random(args.seed + 2).shuffle(user_rec_train)
    random.Random(args.seed + 3).shuffle(user_rec_valid)

    outputs = {
        "stage1_material_train": material_train,
        "stage1_material_valid": material_valid,
        "stage2_user_rec_train": user_rec_train,
        "stage2_user_rec_valid": user_rec_valid,
    }
    for name, records in outputs.items():
        write_jsonl(records, ROOT / "data" / "processed" / f"{name}.jsonl")

    report = {
        "seed": args.seed,
        "user_repeat": args.user_repeat,
        "counts": {name: len(records) for name, records in outputs.items()},
        "stage2_train_tasks": dict(Counter(str(r.get("task_type")) for r in user_rec_train)),
    }
    report_path = ROOT / "data" / "eda" / "STAGED_SFT_REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

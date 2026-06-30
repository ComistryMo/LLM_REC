#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.data_schema import find_itemic_tokens
from llmrec.metrics import domain_hit, exact_match, pass_at_k
from llmrec.utils import read_jsonl


def gold_from_messages(record: dict) -> str:
    for msg in reversed(record.get("messages", [])):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return str(record.get("gold", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local validation metrics for OneReason LLM-Rec tasks.")
    parser.add_argument("--predictions", help="JSONL with id/prediction/gold/domain fields.")
    parser.add_argument("--valid-material", default="data/processed/valid_material.jsonl")
    parser.add_argument("--valid-user", default="data/processed/valid_user.jsonl")
    parser.add_argument("--valid-rec", default="data/processed/valid_rec.jsonl")
    parser.add_argument("--valid-world", default="data/processed/valid_world.jsonl")
    parser.add_argument("--valid-mix", default="data/processed/valid_mix_v1.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.predictions:
        rows = read_jsonl(ROOT / args.predictions)
    else:
        rows = []
        for split_name, path in [
            ("valid_material", args.valid_material),
            ("valid_user", args.valid_user),
            ("valid_rec", args.valid_rec),
            ("valid_world", args.valid_world),
        ]:
            p = ROOT / path
            if p.exists():
                for record in read_jsonl(p):
                    record["split"] = split_name
                    record["prediction"] = record.get("prediction", gold_from_messages(record))
                    record["gold"] = record.get("gold", gold_from_messages(record))
                    rows.append(record)
        if not rows and (ROOT / args.valid_mix).exists():
            for record in read_jsonl(ROOT / args.valid_mix):
                record["split"] = "valid_mix"
                record["prediction"] = record.get("prediction", gold_from_messages(record))
                record["gold"] = record.get("gold", gold_from_messages(record))
                rows.append(record)

    counters: Counter[str] = Counter()
    total = len(rows)
    examples = []
    for row in rows:
        pred = str(row.get("prediction", row.get("pred", "")))
        gold = str(row.get("gold", gold_from_messages(row)))
        domain = str(row.get("domain", "general"))
        if exact_match(pred, gold):
            counters["exact_match"] += 1
        if pass_at_k(pred, gold, k=args.k):
            counters[f"pass@{args.k}"] += 1
        if find_itemic_tokens(pred):
            counters["itemic_valid"] += 1
        if domain_hit(pred, domain) or domain in {"general", "world"}:
            counters["domain_hit"] += 1
        if len(examples) < 10:
            examples.append({"id": row.get("id"), "domain": domain, "pred": pred[:160], "gold": gold[:160]})

    report = {
        "total": total,
        "exact_match": counters["exact_match"] / total if total else 0.0,
        f"pass@{args.k}": counters[f"pass@{args.k}"] / total if total else 0.0,
        "itemic_valid_rate": counters["itemic_valid"] / total if total else 0.0,
        "domain_hit_rate": counters["domain_hit"] / total if total else 0.0,
        "examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

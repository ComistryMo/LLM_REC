#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from llmrec.itemic_training import build_variant, messages_from_record, stable_key, target_itemic


ENHANCED_VARIANTS = ["full"] * 50 + ["sid"] * 20 + ["a"] * 10 + ["b"] * 10 + ["c"] * 10


def reservoir(path: Path, limit: int, seed: int) -> tuple[list[list[dict[str, str]]], Counter[str]]:
    rng = random.Random(seed)
    sample: list[list[dict[str, str]]] = []
    counts: Counter[str] = Counter()
    seen_keys: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                messages = messages_from_record(json.loads(line))
                key = stable_key(messages)
            except Exception:
                counts["invalid"] += 1
                continue
            if key in seen_keys:
                counts["duplicate_itemic"] += 1
                continue
            seen_keys.add(key)
            counts["eligible"] += 1
            if len(sample) < limit:
                sample.append(messages)
            else:
                index = rng.randrange(counts["eligible"])
                if index < limit:
                    sample[index] = messages
    rng.shuffle(sample)
    return sample, counts


def validation_tokens(path: Path | None) -> set[str]:
    if path is None:
        return set()
    result = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            messages = messages_from_record(json.loads(line))
            result.add(target_itemic(messages))
    return result


def write_recipe(path: Path, records: list[list[dict[str, str]]], variants: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as stream:
        for index, messages in enumerate(records):
            variant = variants[index % len(variants)]
            output = build_variant(messages, variant)
            stream.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[variant] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe itemic direct and hierarchy pilot datasets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, source_counts = reservoir(args.input, args.max_items, args.seed)
    held_out = validation_tokens(args.validation)
    train_tokens = {target_itemic(messages) for messages in records}
    overlap = train_tokens & held_out
    if overlap:
        raise SystemExit(f"validation leakage detected: {len(overlap)} itemic tokens")

    outputs = {
        "direct": ("itemic_direct_train.jsonl", ["full"]),
        "sid_only": ("itemic_sid_only_train.jsonl", ["sid"]),
        "enhanced": ("itemic_enhanced_train.jsonl", ENHANCED_VARIANTS),
    }
    recipes = {}
    for name, (filename, variants) in outputs.items():
        path = args.output_dir / filename
        recipes[name] = {"path": str(path), "counts": dict(write_recipe(path, records, variants))}

    manifest = {
        "input": str(args.input),
        "validation": str(args.validation) if args.validation else None,
        "seed": args.seed,
        "requested_items": args.max_items,
        "selected_items": len(records),
        "source_counts": dict(source_counts),
        "validation_itemics": len(held_out),
        "train_validation_overlap": len(overlap),
        "recipes": recipes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

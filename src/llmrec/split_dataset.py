from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .mix_datasets import to_llamafactory_sharegpt
from .utils import read_jsonl, write_jsonl


def content_key(record: dict[str, Any]) -> str:
    payload = json.dumps(record.get("messages", []), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _source_seed(seed: int, source: str) -> int:
    digest = hashlib.sha256(f"{seed}:{source}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def split_official_dataset(
    input_path: str | Path,
    train_path: str | Path,
    valid_path: str | Path,
    report_path: str | Path,
    valid_ratio: float = 0.02,
    seed: int = 2026,
    max_length: int = 8192,
    deduplicate: bool = True,
    llamafactory_prefix: str | Path | None = None,
) -> dict[str, Any]:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")

    records = read_jsonl(input_path)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    excluded_long = 0
    removed_duplicates = 0

    for record in records:
        length = int(record.get("input_len") or 0) + int(record.get("output_len") or 0)
        if max_length > 0 and length > max_length:
            excluded_long += 1
            continue
        key = content_key(record)
        if deduplicate and key in seen:
            removed_duplicates += 1
            continue
        seen.add(key)
        by_source[str(record.get("source") or "unknown")].append(record)

    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}
    for source, source_records in sorted(by_source.items()):
        rng = random.Random(_source_seed(seed, source))
        rng.shuffle(source_records)
        valid_count = max(1, round(len(source_records) * valid_ratio)) if len(source_records) > 1 else 0
        valid.extend(source_records[:valid_count])
        train.extend(source_records[valid_count:])
        split_counts[source] = {
            "train": len(source_records) - valid_count,
            "valid": valid_count,
        }

    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(valid)
    write_jsonl(train, train_path)
    write_jsonl(valid, valid_path)

    if llamafactory_prefix:
        prefix = Path(llamafactory_prefix)
        write_jsonl(
            (to_llamafactory_sharegpt(record) for record in train),
            prefix.with_name(prefix.name + "_train_sharegpt.jsonl"),
        )
        write_jsonl(
            (to_llamafactory_sharegpt(record) for record in valid),
            prefix.with_name(prefix.name + "_valid_sharegpt.jsonl"),
        )

    report = {
        "input": str(input_path),
        "original": len(records),
        "train": len(train),
        "valid": len(valid),
        "excluded_over_max_length": excluded_long,
        "removed_exact_duplicates": removed_duplicates,
        "max_length": max_length,
        "valid_ratio": valid_ratio,
        "seed": seed,
        "deduplicate": deduplicate,
        "by_source": split_counts,
        "train_tasks": dict(Counter(str(r.get("task_type")) for r in train)),
        "valid_tasks": dict(Counter(str(r.get("task_type")) for r in valid)),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

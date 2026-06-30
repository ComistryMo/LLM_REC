from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import load_yaml, read_jsonl, stable_shuffle, write_jsonl

DEFAULT_RATIOS = {"material": 0.25, "user": 0.25, "rec": 0.30, "world": 0.20}

TASK_TO_BUCKET = {
    "material_understanding": "material",
    "itemic_grounding": "material",
    "item_qa": "material",
    "user_demand_understanding": "user",
    "recommendation": "rec",
    "world_knowledge": "world",
}


def bucket_record(record: dict[str, Any]) -> str:
    if record.get("bucket") in DEFAULT_RATIOS:
        return str(record["bucket"])
    return TASK_TO_BUCKET.get(str(record.get("task_type", "other")), "world")


def to_llamafactory_sharegpt(record: dict[str, Any]) -> dict[str, Any]:
    conversations = []
    for msg in record.get("messages", []):
        role = msg.get("role")
        if role == "system":
            conversations.append({"from": "system", "value": msg.get("content", "")})
        elif role == "user":
            conversations.append({"from": "human", "value": msg.get("content", "")})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": msg.get("content", "")})
    return {
        "id": record.get("id"),
        "conversations": conversations,
        "task_type": record.get("task_type"),
        "domain": record.get("domain"),
    }


def mix_datasets(
    inputs: list[str | Path],
    output_train: str | Path,
    output_valid: str | Path,
    ratio_config: str | Path | None = None,
    valid_ratio: float = 0.02,
    seed: int = 2026,
    output_llamafactory_prefix: str | Path | None = None,
) -> tuple[int, int]:
    ratios = dict(DEFAULT_RATIOS)
    if ratio_config:
        ratios.update({k: float(v) for k, v in load_yaml(ratio_config).items() if k in ratios})
    total_ratio = sum(ratios.values())
    ratios = {k: v / total_ratio for k, v in ratios.items()}

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in inputs:
        if not Path(path).exists():
            continue
        for record in read_jsonl(path):
            buckets[bucket_record(record)].append(record)
    for key in ratios:
        buckets[key] = stable_shuffle(buckets[key], seed + len(key))

    available = {k: len(v) for k, v in buckets.items()}
    if not any(available.values()):
        raise ValueError("No records found for mixing.")
    max_total = min(
        int(available[k] / ratio) for k, ratio in ratios.items() if ratio > 0 and available[k] > 0
    )
    if max_total <= 0:
        max_total = sum(available.values())

    mixed: list[dict[str, Any]] = []
    for bucket, ratio in ratios.items():
        take = min(len(buckets[bucket]), max(0, round(max_total * ratio)))
        mixed.extend(buckets[bucket][:take])
    mixed = stable_shuffle(mixed, seed)
    split = max(1, int(len(mixed) * (1 - valid_ratio))) if len(mixed) > 1 else len(mixed)
    train, valid = mixed[:split], mixed[split:]
    train_count = write_jsonl(train, output_train)
    valid_count = write_jsonl(valid, output_valid)
    if output_llamafactory_prefix:
        prefix = Path(output_llamafactory_prefix)
        write_jsonl([to_llamafactory_sharegpt(r) for r in train], prefix.with_name(prefix.name + "_train_sharegpt.jsonl"))
        write_jsonl([to_llamafactory_sharegpt(r) for r in valid], prefix.with_name(prefix.name + "_valid_sharegpt.jsonl"))
    return train_count, valid_count


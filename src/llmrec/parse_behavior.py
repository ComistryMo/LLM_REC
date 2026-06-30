from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .data_schema import infer_domain_from_text, normalize_itemic_text
from .utils import load_yaml, read_records, write_jsonl

DEFAULT_ALIASES = {
    "user_id": ["user_id", "uid", "user", "userid"],
    "timestamp": ["timestamp", "time", "ts", "event_time", "datetime"],
    "action_type": ["action_type", "action", "event", "behavior", "type"],
    "domain": ["domain", "item_domain", "scene", "biz_type"],
    "itemic_pattern": ["itemic_pattern", "itemic", "itemic_token", "item_token", "item"],
    "query": ["query", "search_query", "keyword", "text"],
    "item_id": ["item_id", "photo_id", "product_id", "ad_id", "live_id"],
}


def _first(record: dict[str, Any], aliases: list[str], default: str = "") -> str:
    for key in aliases:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def load_aliases(path: str | Path | None) -> dict[str, list[str]]:
    aliases = {k: list(v) for k, v in DEFAULT_ALIASES.items()}
    if path:
        config = load_yaml(path)
        for key, value in config.get("aliases", config).items():
            aliases[key] = value if isinstance(value, list) else [str(value)]
    return aliases


def normalize_event(record: dict[str, Any], aliases: dict[str, list[str]]) -> dict[str, Any]:
    itemic = normalize_itemic_text(_first(record, aliases["itemic_pattern"]))
    query = _first(record, aliases["query"])
    domain = _first(record, aliases["domain"])
    if not domain:
        domain = infer_domain_from_text(f"{itemic} {query}")
    event = {
        "user_id": _first(record, aliases["user_id"]),
        "timestamp": _first(record, aliases["timestamp"]),
        "action_type": _first(record, aliases["action_type"], "unknown"),
        "domain": domain,
        "itemic_pattern": itemic,
        "query": query,
        "item_id": _first(record, aliases["item_id"]),
    }
    extras = {
        k: v
        for k, v in record.items()
        if k not in {alias for values in aliases.values() for alias in values} and v not in (None, "")
    }
    if extras:
        event["extra"] = extras
    return event


def parse_behavior_files(
    inputs: list[str | Path],
    output_sequences: str | Path,
    output_events: str | Path | None = None,
    alias_config: str | Path | None = None,
) -> tuple[int, int]:
    aliases = load_aliases(alias_config)
    events: list[dict[str, Any]] = []
    for path in inputs:
        for record in read_records(path):
            event = normalize_event(record, aliases)
            if event["user_id"] and event["timestamp"]:
                events.append(event)
    events.sort(key=lambda e: (e["user_id"], e["timestamp"], e.get("item_id", ""), e.get("itemic_pattern", "")))
    if output_events:
        write_jsonl(events, output_events)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["user_id"]].append({k: v for k, v in event.items() if k != "user_id"})

    sequences = [
        {"user_id": user_id, "sequence_len": len(seq), "events": seq}
        for user_id, seq in sorted(grouped.items(), key=lambda x: x[0])
    ]
    return write_jsonl(sequences, output_sequences), len(events)


from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data_schema import BEGIN_RE, ITEMIC_RE, SECRET_RE, find_itemic_tokens, message_text, normalize_itemic_text
from .utils import read_jsonl


def validate_messages(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["messages_missing"]
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if any(role not in {"system", "user", "assistant"} for role in roles):
        errors.append("bad_role")
    if roles and roles[-1] != "assistant":
        errors.append("last_message_not_assistant")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or not str(msg.get("content", "")).strip():
            errors.append(f"empty_message_{i}")
    if any(m.get("role") == "assistant" and not str(m.get("content", "")).strip() for m in messages if isinstance(m, dict)):
        errors.append("empty_assistant")
    return errors


def invalid_itemic_fragments(text: str) -> list[str]:
    text = normalize_itemic_text(text)
    valid_spans = [m.span() for m in ITEMIC_RE.finditer(text)]
    fragments: list[str] = []
    for match in BEGIN_RE.finditer(text):
        if not any(start <= match.start() < end for start, end in valid_spans):
            fragments.append(match.group(0))
    return fragments


def check_records(path: str | Path, max_length: int = 8192) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    bad_examples: list[dict[str, Any]] = []
    rows = read_jsonl(path)
    for lineno, record in enumerate(rows, 1):
        errors = validate_messages(record)
        text = message_text(record.get("messages", []))
        if SECRET_RE.search(text) or SECRET_RE.search(json.dumps(record, ensure_ascii=False)):
            errors.append("secret_like_text")
        measured_length = int(record.get("input_len") or 0) + int(record.get("output_len") or 0)
        if measured_length > max_length or (not measured_length and len(text.split()) > max_length):
            errors.append("too_long")
        if invalid_itemic_fragments(text):
            errors.append("invalid_itemic_fragment")
        if record.get("has_itemic") and not find_itemic_tokens(text):
            errors.append("has_itemic_but_no_valid_token")
        if errors:
            for error in errors:
                counters[error] += 1
            if len(bad_examples) < 20:
                bad_examples.append({"line": lineno, "id": record.get("id"), "errors": errors})
    return {
        "path": str(path),
        "total": len(rows),
        "error_counts": dict(counters),
        "bad_examples": bad_examples,
        "ok": not counters,
    }

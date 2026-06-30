from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def repo_root_from(path: str | Path) -> Path:
    path = Path(path).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = _load_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [x.strip().strip("\"'") for x in inner.split(",") if x.strip()] if inner else []
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return float(value) if "." in value else int(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value.strip("\"'")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        if indent == 0:
            parsed = _parse_scalar(value)
            root[key] = parsed
            current = parsed if isinstance(parsed, dict) else None
        elif current is not None:
            current[key] = _parse_scalar(value)
    return root


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            values = obj if isinstance(obj, list) else [obj]
            for item_index, value in enumerate(values):
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL value must be an object at {path}:{lineno}[{item_index}]"
                    )
                yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jsonlines"}:
        yield from iter_jsonl(path)
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from (x for x in data if isinstance(x, dict))
            return
        if isinstance(data, dict):
            for key in ["data", "records", "samples", "items"]:
                if isinstance(data.get(key), list):
                    yield from (x for x in data[key] if isinstance(x, dict))
                    return
            yield data
            return
        raise ValueError(f"Unsupported JSON root in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
        return
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    raise ValueError(f"Unsupported input format: {path}")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_records(path))


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def stable_shuffle(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    return records


def percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return float(values[lower])
    frac = pos - lower
    return float(values[lower] * (1 - frac) + values[upper] * frac)

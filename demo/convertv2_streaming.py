#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from glob import glob
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

import convertv2 as official


def discover(patterns: list[str]) -> list[Path]:
    result = []
    for pattern in patterns:
        if "*" in pattern:
            result.extend(Path(value) for value in sorted(glob(pattern, recursive=True)))
        else:
            path = Path(pattern)
            result.extend(sorted(path.rglob("*.parquet")) if path.is_dir() else [path])
    return [path for path in result if path.suffix == ".parquet"]


def rows(path: Path, batch_size: int) -> Iterable[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(path)
    wanted = [name for name in ("messages", "uuid", "source", "line_id") if name in parquet.schema_arrow.names]
    offset = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=wanted):
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            yield offset + index, {name: values[index] for name, values in columns.items()}
        offset += batch.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory streaming variant of the official convertv2.py.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--filter-log")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-token-types", type=int, default=3)
    parser.add_argument("--no-filter-sid-tokens", dest="filter_sid_tokens", action="store_false")
    parser.add_argument("--no-add-think-pattern", dest="add_think_pattern", action="store_false")
    args = parser.parse_args()

    files = discover(args.input)
    if not files:
        raise SystemExit("no parquet files found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    filter_stream = Path(args.filter_log).open("w", encoding="utf-8") if args.filter_log else None

    with output.open("w", encoding="utf-8") as target:
        for path in files:
            parquet = pq.ParquetFile(path)
            if "messages" not in parquet.schema_arrow.names:
                stats["files:skip_no_messages_column"] += 1
                stats["rows:skip_no_messages_column"] += parquet.metadata.num_rows
                continue
            for row_index, row in rows(path, args.batch_size):
                stats["rows:seen"] += 1
                raw = row.get("messages")
                if raw is None:
                    stats["skip:no_messages"] += 1
                    continue
                try:
                    messages = json.loads(raw) if isinstance(raw, str) else raw
                    token_hits: dict[str, int] = {}
                    think_events: list[str] = []
                    converted = official.convert_messages(
                        messages,
                        add_think_pattern=args.add_think_pattern,
                        do_filter_sid=args.filter_sid_tokens,
                        stats=stats,
                        row_token_hits=token_hits,
                        row_think_events=think_events,
                    )
                    text = "".join(message["content"] for message in converted)
                    valid, found = official.check_itemic_token_types(text, args.max_token_types)
                    if not valid:
                        stats["dropped:itemic_overflow"] += 1
                        if filter_stream:
                            filter_stream.write(json.dumps({"file": str(path), "row_idx": row_index, "reason": "drop:itemic_overflow", "itemic_letters_found": sorted(found)}, ensure_ascii=False) + "\n")
                        continue
                    record = official.to_alpaca(converted)
                    if record is None:
                        stats["skip:to_alpaca_empty"] += 1
                        continue
                    target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats["records:written"] += 1
                except Exception as error:
                    stats["skip:exception"] += 1
                    if filter_stream:
                        filter_stream.write(json.dumps({"file": str(path), "row_idx": row_index, "reason": "skip:exception", "error": repr(error)}, ensure_ascii=False) + "\n")
            stats["files:processed"] += 1

    if filter_stream:
        filter_stream.close()
    report = {
        "input": args.input,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "stats": dict(stats),
        "note": "Records are streamed in deterministic file order; trainer-side shuffling remains enabled.",
    }
    if args.summary:
        Path(args.summary).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

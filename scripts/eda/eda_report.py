#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.data_schema import find_itemic_tokens, itemic_domains
from llmrec.utils import ensure_dir, percentile, read_jsonl


def counter_md(title: str, counter: Counter, limit: int = 30) -> list[str]:
    lines = [f"## {title}", "", "| value | count |", "|---|---:|"]
    for key, value in counter.most_common(limit):
        lines.append(f"| {key or '(empty)'} | {value} |")
    if not counter:
        lines.append("| (none) | 0 |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown EDA report.")
    parser.add_argument("--sft", default="data/processed/sft_unified.jsonl")
    parser.add_argument("--sequences", default="data/processed/user_sequences.jsonl")
    parser.add_argument("--output", default="data/eda/EDA_REPORT.md")
    args = parser.parse_args()

    sft_path = ROOT / args.sft
    seq_path = ROOT / args.sequences
    records = read_jsonl(sft_path) if sft_path.exists() else []
    sequences = read_jsonl(seq_path) if seq_path.exists() else []

    task_counter = Counter(r.get("task_type", "") for r in records)
    output_counter = Counter(r.get("output_type", "") for r in records)
    domain_counter = Counter(r.get("domain", "") for r in records)
    action_counter: Counter[str] = Counter()
    seq_lens: list[int] = []
    input_lens = [int(r.get("input_len") or 0) for r in records]
    output_lens = [int(r.get("output_len") or 0) for r in records]
    token_counter: Counter[str] = Counter()
    null_rows = 0
    invalid_itemic_rows = 0

    for record in records:
        text = "\n".join(str(m.get("content", "")) for m in record.get("messages", []) if isinstance(m, dict))
        tokens = find_itemic_tokens(text)
        token_counter.update(tokens)
        if record.get("has_itemic") and not tokens:
            invalid_itemic_rows += 1
        if not record.get("messages") or any(not str(v).strip() for v in [record.get("id", "")]):
            null_rows += 1
        detected_domains = itemic_domains(text)
        if len(detected_domains) > 1:
            domain_counter["mixed_itemic_detected"] += 1

    for seq in sequences:
        events = seq.get("events") or []
        seq_lens.append(len(events))
        for event in events:
            action_counter[str(event.get("action_type", ""))] += 1

    total_itemic_rows = sum(1 for r in records if r.get("has_itemic"))
    legal_rate = 1.0 if total_itemic_rows == 0 else (total_itemic_rows - invalid_itemic_rows) / total_itemic_rows

    lines = [
        "# EDA Report",
        "",
        f"- SFT samples: {len(records)}",
        f"- User sequences: {len(sequences)}",
        f"- Itemic token legal rate: {legal_rate:.4f}",
        f"- Null/abnormal rows: {null_rows}",
        f"- Invalid itemic rows: {invalid_itemic_rows}",
        "",
        "## User Sequence Length",
        "",
        f"- P50: {percentile(seq_lens, 0.50):.2f}",
        f"- P90: {percentile(seq_lens, 0.90):.2f}",
        f"- P99: {percentile(seq_lens, 0.99):.2f}",
        "",
        "## Input / Output Length",
        "",
        f"- input P50/P90/P99: {percentile(input_lens, 0.50):.2f} / {percentile(input_lens, 0.90):.2f} / {percentile(input_lens, 0.99):.2f}",
        f"- output P50/P90/P99: {percentile(output_lens, 0.50):.2f} / {percentile(output_lens, 0.90):.2f} / {percentile(output_lens, 0.99):.2f}",
        "",
    ]
    lines += counter_md("Task Type Distribution", task_counter)
    lines += counter_md("Output Type Distribution", output_counter)
    lines += counter_md("Domain Distribution", domain_counter)
    lines += counter_md("Action Type Distribution", action_counter)
    lines += counter_md("Top Itemic Tokens", token_counter, limit=50)

    output = ROOT / args.output
    ensure_dir(output.parent)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.data_schema import find_itemic_tokens, itemic_domains
from llmrec.format_check import invalid_itemic_fragments
from llmrec.utils import ensure_dir, percentile, read_jsonl


def counter_md(title: str, counter: Counter, limit: int = 30) -> list[str]:
    lines = [f"## {title}", "", "| value | count |", "|---|---:|"]
    for key, value in counter.most_common(limit):
        lines.append(f"| {key or '(empty)'} | {value} |")
    if not counter:
        lines.append("| (none) | 0 |")
    lines.append("")
    return lines


def cross_md(counter: Counter) -> list[str]:
    lines = ["## Source / Task Cross-Tab", "", "| source | task | count |", "|---|---|---:|"]
    for (source, task), count in sorted(counter.items()):
        lines.append(f"| {source or '(empty)'} | {task or '(empty)'} | {count} |")
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
    source_counter = Counter(r.get("source", "") for r in records)
    source_task_counter = Counter((r.get("source", ""), r.get("task_type", "")) for r in records)
    action_counter: Counter[str] = Counter()
    atomic_action_counter: Counter[str] = Counter()
    seq_lens: list[int] = []
    input_lens = [int(r.get("input_len") or 0) for r in records]
    output_lens = [int(r.get("output_len") or 0) for r in records]
    token_counter: Counter[str] = Counter()
    null_rows = 0
    invalid_itemic_rows = 0
    mixed_itemic_rows = 0
    prompt_history_count = 0
    over_4096 = 0
    over_8192 = 0
    sample_hashes: Counter[str] = Counter()

    action_pattern = re.compile(r"\[(?:直播|商品|广告|视频)-([^\]]+)\]")

    for record in records:
        text = "\n".join(str(m.get("content", "")) for m in record.get("messages", []) if isinstance(m, dict))
        tokens = find_itemic_tokens(text)
        token_counter.update(tokens)
        sample_hashes[hashlib.sha1(json.dumps(record.get("messages", []), ensure_ascii=False).encode()).hexdigest()] += 1
        total_length = int(record.get("input_len") or 0) + int(record.get("output_len") or 0)
        over_4096 += total_length > 4096
        over_8192 += total_length > 8192
        if invalid_itemic_fragments(text) or (record.get("has_itemic") and not tokens):
            invalid_itemic_rows += 1
        if not record.get("messages") or any(not str(v).strip() for v in [record.get("id", "")]):
            null_rows += 1
        detected_domains = itemic_domains(text)
        if len(detected_domains) > 1:
            mixed_itemic_rows += 1
        if str(record.get("source", "")).startswith("懂用户"):
            actions = action_pattern.findall(text)
            if actions:
                prompt_history_count += 1
                seq_lens.append(len(actions))
                action_counter.update(actions)
                for action in actions:
                    atomic_action_counter.update(part for part in action.split("/") if part)

    for seq in sequences:
        events = seq.get("events") or []
        seq_lens.append(len(events))
        for event in events:
            action_counter[str(event.get("action_type", ""))] += 1

    total_itemic_rows = sum(1 for r in records if r.get("has_itemic"))
    legal_rate = 1.0 if total_itemic_rows == 0 else (total_itemic_rows - invalid_itemic_rows) / total_itemic_rows
    duplicate_rows = sum(count - 1 for count in sample_hashes.values() if count > 1)
    duplicate_groups = sum(count > 1 for count in sample_hashes.values())

    lines = [
        "# EDA Report",
        "",
        f"- SFT samples: {len(records)}",
        f"- User sequences: {len(sequences)}",
        f"- Prompt-derived user histories: {prompt_history_count}",
        f"- Itemic token legal rate: {legal_rate:.4f}",
        f"- Null/abnormal rows: {null_rows}",
        f"- Invalid itemic rows: {invalid_itemic_rows}",
        f"- Mixed-domain itemic rows: {mixed_itemic_rows}",
        f"- Samples over 4096 tokens: {over_4096}",
        f"- Samples over 8192 tokens: {over_8192}",
        f"- Duplicate rows/groups: {duplicate_rows} / {duplicate_groups}",
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
    lines += counter_md("Source Distribution", source_counter)
    lines += cross_md(source_task_counter)
    lines += counter_md("Output Type Distribution", output_counter)
    lines += counter_md("Domain Distribution", domain_counter)
    lines += counter_md("Raw Action Label Distribution", action_counter)
    lines += counter_md("Atomic Action Type Distribution", atomic_action_counter)
    lines += counter_md("Top Itemic Tokens", token_counter, limit=50)

    output = ROOT / args.output
    ensure_dir(output.parent)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

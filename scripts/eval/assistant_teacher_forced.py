#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmrec.itemic_training import messages_from_record


def token_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def stable_key(record: dict[str, Any]) -> str:
    value = str(record.get("id", json.dumps(record, ensure_ascii=False, sort_keys=True)))
    return hashlib.sha256(value.encode()).hexdigest()


def prepare_examples(
    path: Path, tokenizer: Any, max_length: int, max_per_task: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            groups[str(record.get("task_type", "unknown"))].append(record)

    examples = []
    stats: dict[str, Any] = {"rows": sum(map(len, groups.values())), "tasks": {}, "skipped": 0}
    for task, records in sorted(groups.items()):
        selected = sorted(records, key=stable_key)[:max_per_task]
        used = 0
        for record in selected:
            messages = messages_from_record(record)
            if not messages or messages[-1].get("role") != "assistant" or not messages[-1].get("content"):
                stats["skipped"] += 1
                continue
            full = token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
            prompt = token_ids(
                tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
            )
            prefix = 0
            for left, right in zip(full, prompt):
                if left != right:
                    break
                prefix += 1
            full = full[:max_length]
            if prefix < 1 or prefix >= len(full):
                stats["skipped"] += 1
                continue
            examples.append({"ids": full, "start": prefix, "task": task})
            used += 1
        stats["tasks"][task] = {"available": len(records), "selected": len(selected), "usable": used}
    return examples, stats


def evaluate(model_path: str, examples: list[dict[str, Any]], tokenizer: Any, device: str) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"nll": 0.0, "correct": 0, "tokens": 0})
    with torch.inference_mode():
        for example in examples:
            ids = torch.tensor([example["ids"]], dtype=torch.long, device=device)
            logits = model(input_ids=ids, use_cache=False).logits[0]
            positions = torch.arange(example["start"], ids.shape[1], device=device)
            selected = logits[positions - 1].float()
            targets = ids[0, positions]
            losses = torch.logsumexp(selected, dim=-1) - selected.gather(1, targets[:, None]).squeeze(1)
            total = totals[example["task"]]
            total["nll"] += losses.sum().item()
            total["correct"] += selected.argmax(-1).eq(targets).sum().item()
            total["tokens"] += targets.numel()
            del ids, logits, selected, targets, losses

    task_results = {}
    aggregate = {"nll": 0.0, "correct": 0, "tokens": 0}
    for task, total in sorted(totals.items()):
        count = int(total["tokens"])
        loss = total["nll"] / count
        task_results[task] = {
            "tokens": count,
            "loss": loss,
            "ppl": math.exp(min(20, loss)),
            "token_accuracy": total["correct"] / count,
        }
        for key in aggregate:
            aggregate[key] += total[key]
    count = int(aggregate["tokens"])
    result = {
        "model": model_path,
        "tasks": task_results,
        "overall": {
            "tokens": count,
            "loss": aggregate["nll"] / count,
            "ppl": math.exp(min(20, aggregate["nll"] / count)),
            "token_accuracy": aggregate["correct"] / count,
        },
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare assistant-only teacher-forced loss by task.")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH; repeat for comparisons")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-per-task", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = dict(item.split("=", 1) for item in args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or next(iter(models.values())), trust_remote_code=True)
    examples, data_stats = prepare_examples(args.validation, tokenizer, args.max_length, args.max_per_task)
    if not examples:
        raise SystemExit("no usable assistant targets")
    payload = {"validation": str(args.validation), "data": data_stats, "models": {}}
    for name, path in models.items():
        payload["models"][name] = evaluate(path, examples, tokenizer, args.device)
        print(name, json.dumps(payload["models"][name], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

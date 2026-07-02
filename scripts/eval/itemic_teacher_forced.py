#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmrec.itemic_training import messages_from_record, split_itemic, target_itemic


POSITIONS = ("domain", "a", "b", "c")


def token_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def locate_last(values: list[int], needle: list[int]) -> int:
    for index in range(len(values) - len(needle), -1, -1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def prepare_examples(path: Path, tokenizer: Any, max_length: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples = []
    stats = {"rows": 0, "missing_itemic": 0, "truncated_itemic": 0, "usable": 0}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stats["rows"] += 1
            messages = messages_from_record(json.loads(line))
            try:
                target = target_itemic(messages)
            except ValueError:
                stats["missing_itemic"] += 1
                continue
            parts = split_itemic(target)
            expected = [tokenizer.convert_tokens_to_ids(part) for part in parts]
            ids = token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
            start = locate_last(ids, expected)
            if start < 1:
                stats["missing_itemic"] += 1
                continue
            ids = ids[:max_length]
            if start + 4 > len(ids):
                stats["truncated_itemic"] += 1
                continue
            examples.append({"ids": ids, "start": start})
    stats["usable"] = len(examples)
    return examples, stats


def evaluate(model_path: str, examples: list[dict[str, Any]], tokenizer: Any, device: str, batch_size: int) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    totals = {name: {"nll": 0.0, "top1": 0, "top5": 0} for name in POSITIONS}
    full_exact = abc_exact = abc_top5_exact = 0

    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset : offset + batch_size]
            length = max(len(example["ids"]) for example in batch)
            ids = torch.full((len(batch), length), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            for row, example in enumerate(batch):
                size = len(example["ids"])
                ids[row, :size] = torch.tensor(example["ids"], device=device)
                attention[row, :size] = 1
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
            for row, example in enumerate(batch):
                positions = torch.arange(example["start"], example["start"] + 4, device=device)
                selected = logits[row, positions - 1].float()
                targets = ids[row, positions]
                predictions = selected.argmax(-1)
                top5 = selected.topk(5, dim=-1).indices
                losses = torch.logsumexp(selected, dim=-1) - selected.gather(1, targets[:, None]).squeeze(1)
                top1_flags = predictions.eq(targets)
                top5_flags = top5.eq(targets[:, None]).any(dim=-1)
                for index, name in enumerate(POSITIONS):
                    totals[name]["nll"] += losses[index].item()
                    totals[name]["top1"] += int(top1_flags[index])
                    totals[name]["top5"] += int(top5_flags[index])
                full_exact += int(top1_flags.all())
                abc_exact += int(top1_flags[1:].all())
                abc_top5_exact += int(top5_flags[1:].all())
            del logits, ids, attention

    count = len(examples)
    positions_result = {
        name: {
            "loss": totals[name]["nll"] / count,
            "ppl": math.exp(min(20, totals[name]["nll"] / count)),
            "top1_acc": totals[name]["top1"] / count,
            "top5_acc": totals[name]["top5"] / count,
        }
        for name in POSITIONS
    }
    result = {
        "model": model_path,
        "samples": count,
        "positions": positions_result,
        "abc_loss": sum(totals[name]["nll"] for name in POSITIONS[1:]) / (3 * count),
        "abc_top1_acc": sum(totals[name]["top1"] for name in POSITIONS[1:]) / (3 * count),
        "abc_top5_acc": sum(totals[name]["top5"] for name in POSITIONS[1:]) / (3 * count),
        "abc_exact_rate": abc_exact / count,
        "abc_top5_exact_rate": abc_top5_exact / count,
        "full_exact_rate": full_exact / count,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced itemic domain/a/b/c evaluation.")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH; repeat for comparisons")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = dict(item.split("=", 1) for item in args.model)
    tokenizer_path = args.tokenizer or next(iter(models.values()))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples, data_stats = prepare_examples(args.validation, tokenizer, args.max_length)
    if not examples:
        raise SystemExit("no usable itemic validation examples")
    result = {"validation": str(args.validation), "data": data_stats, "models": {}}
    for name, path in models.items():
        result["models"][name] = evaluate(path, examples, tokenizer, args.device, args.batch_size)
        print(name, json.dumps(result["models"][name], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

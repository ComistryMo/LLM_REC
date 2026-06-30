#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.parse_sft import assistant_text, user_text
from llmrec.data_schema import find_itemic_tokens
from llmrec.utils import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Left-crop SFT user content while preserving assistant targets.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-length", type=int, default=3800)
    parser.add_argument(
        "--drop-over-target",
        action="store_true",
        help="Drop records that still exceed target length after preserving the answer and minimum context.",
    )
    args = parser.parse_args()
    if len(args.inputs) != len(args.outputs):
        raise ValueError("inputs and outputs must have the same length")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    summary = {}
    for input_value, output_value in zip(args.inputs, args.outputs):
        input_path = ROOT / input_value
        output_path = ROOT / output_value
        cropped_count = 0
        dropped_count = 0

        def records():
            nonlocal cropped_count, dropped_count
            for record in iter_jsonl(input_path):
                messages = [dict(message) for message in record.get("messages", [])]
                prompt = user_text(messages)
                answer = assistant_text(messages)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                original_total = len(prompt_ids) + len(answer_ids)
                if original_total > args.target_length:
                    user_indexes = [i for i, message in enumerate(messages) if message.get("role") == "user"]
                    if not user_indexes:
                        continue
                    index = user_indexes[-1]
                    content = str(messages[index].get("content", ""))
                    content_ids = tokenizer.encode(content, add_special_tokens=False)
                    keep = max(128, len(content_ids) - (original_total - args.target_length) - 32)
                    cropped = tokenizer.decode(content_ids[-keep:], skip_special_tokens=False)
                    if "\n" in cropped:
                        cropped = cropped.split("\n", 1)[1]
                    cropped = re.sub(r"^(?:\s*<s_[abc]_[^>]+>)+", "", cropped).lstrip()
                    messages[index]["content"] = cropped
                    cropped_count += 1
                prompt = user_text(messages)
                answer = assistant_text(messages)
                result = dict(record)
                result["messages"] = messages
                itemic_tokens = find_itemic_tokens("\n".join(str(message.get("content", "")) for message in messages))
                result["has_itemic"] = bool(itemic_tokens)
                result["itemic_count"] = len(itemic_tokens)
                result["input_len"] = len(tokenizer.encode(prompt, add_special_tokens=False))
                result["output_len"] = len(tokenizer.encode(answer, add_special_tokens=False))
                if args.drop_over_target and result["input_len"] + result["output_len"] > args.target_length:
                    dropped_count += 1
                    continue
                result["was_length_cropped"] = original_total > args.target_length
                result["original_total_len"] = original_total
                yield result

        count = write_jsonl(records(), output_path)
        summary[str(output_path)] = {
            "records": count,
            "cropped": cropped_count,
            "dropped": dropped_count,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

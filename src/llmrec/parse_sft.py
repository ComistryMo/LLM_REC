from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from .data_schema import find_itemic_tokens, message_text, normalize_itemic_text, token_len
from .task_router import infer_domain, infer_output_type, infer_task_type
from .utils import iter_records, write_jsonl

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "instruction": "user",
    "prompt": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "response": "assistant",
    "completion": "assistant",
    "system": "system",
}


def normalize_role(role: Any) -> str:
    return ROLE_MAP.get(str(role or "").lower(), str(role or "user").lower())


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        messages = []
        for msg in record["messages"]:
            if not isinstance(msg, dict):
                continue
            role = normalize_role(msg.get("role") or msg.get("from"))
            content = normalize_itemic_text(str(msg.get("content") or msg.get("value") or ""))
            if content:
                messages.append({"role": role, "content": content})
        if messages:
            return messages

    if isinstance(record.get("conversations"), list):
        return normalize_messages({"messages": record["conversations"]})

    system = record.get("system") or record.get("system_prompt")
    instruction = record.get("instruction") or record.get("prompt") or record.get("query") or record.get("question")
    input_text = record.get("input") or record.get("context") or ""
    output = record.get("output") or record.get("response") or record.get("answer") or record.get("completion")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": normalize_itemic_text(str(system))})
    if instruction is not None:
        prompt = str(instruction)
        if input_text:
            prompt = f"{prompt}\n{input_text}"
        messages.append({"role": "user", "content": normalize_itemic_text(prompt)})
    if output is not None:
        messages.append({"role": "assistant", "content": normalize_itemic_text(str(output))})
    return messages


def assistant_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages if m.get("role") == "assistant")


def user_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages if m.get("role") in {"system", "user"})


def convert_record(
    record: dict[str, Any], idx: int, length_fn: Callable[[str], int] = token_len
) -> dict[str, Any]:
    messages = normalize_messages(record)
    prompt = user_text(messages)
    output = assistant_text(messages)
    all_text = message_text(messages)
    itemic_tokens = find_itemic_tokens(all_text)
    source = str(record.get("source") or record.get("dataset") or "sft")
    digest = hashlib.sha1(f"{source}\0{prompt}\0{output}".encode("utf-8")).hexdigest()[:12]
    record_id = record.get("id") or record.get("sample_id") or f"sft-{idx:08d}-{digest}"
    task_type = record.get("task_type") or infer_task_type(f"{source}\n{prompt}", output)
    output_type = record.get("output_type") or infer_output_type(output)
    domain = record.get("domain") or infer_domain(prompt, output)
    return {
        "id": str(record_id),
        "messages": messages,
        "task_type": str(task_type),
        "output_type": str(output_type),
        "domain": str(domain),
        "has_itemic": bool(itemic_tokens),
        "has_cot": any(k in output for k in ["思考", "推理", "分析", "因为", "因此", "<think>"]),
        "input_len": length_fn(prompt),
        "output_len": length_fn(output),
        "itemic_count": len(itemic_tokens),
        "source": source,
    }


def parse_sft_files(
    inputs: list[str | Path],
    output: str | Path,
    length_fn: Callable[[str], int] = token_len,
) -> int:
    idx = 0

    def converted_records():
        nonlocal idx
        for path_value in inputs:
            path = Path(path_value)
            for record in iter_records(path):
                idx += 1
                if not record.get("source") and not record.get("dataset"):
                    record = {**record, "source": path.stem}
                yield convert_record(record, idx, length_fn=length_fn)

    return write_jsonl(converted_records(), output)

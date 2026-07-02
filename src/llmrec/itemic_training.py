from __future__ import annotations

import hashlib
import re
from typing import Any

from .data_schema import ITEMIC_RE, normalize_itemic_text


CONTROL_RE = re.compile(r"/(?:no_)?think\s*$")
CODE_RE = re.compile(r"<s_([abc])_([A-Za-z0-9_-]+)>")


def split_itemic(token: str) -> tuple[str, str, str, str]:
    normalized = normalize_itemic_text(token).strip()
    match = ITEMIC_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(f"invalid itemic token: {token!r}")
    codes = CODE_RE.findall(match.group("codes"))
    if [level for level, _ in codes] != ["a", "b", "c"]:
        raise ValueError(f"invalid itemic hierarchy: {token!r}")
    return match.group("begin"), *(f"<s_{level}_{value}>" for level, value in codes)


def messages_from_record(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    result = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message is not an object")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role and content:
            result.append({"role": role, "content": content})
    if not result:
        raise ValueError("record has no usable messages")
    return result


def target_itemic(messages: list[dict[str, str]]) -> str:
    assistant = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
    matches = list(ITEMIC_RE.finditer(normalize_itemic_text(assistant)))
    if not matches:
        raise ValueError("assistant response has no complete itemic token")
    return matches[-1].group(0)


def source_prompt(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in reversed(messages) if m["role"] in {"user", "human"}), "")
    if not user:
        raise ValueError("record has no user prompt")
    return system, CONTROL_RE.sub("", user).rstrip()


def build_variant(messages: list[dict[str, str]], variant: str) -> dict[str, Any]:
    token = target_itemic(messages)
    begin, code_a, code_b, code_c = split_itemic(token)
    system, prompt = source_prompt(messages)

    if variant == "full":
        instruction = system
        input_text = f"{prompt}/no_think"
        output = token
    elif variant == "sid":
        instruction = "你是精准的 itemic SID 编码器。评测器已经给定物料域前缀，请只输出三个 SID token。"
        input_text = (
            f"{prompt}\n\n已知输出域前缀为 {begin}，请只输出其后的 <s_a_...><s_b_...><s_c_...>，"
            "不要重复域前缀，不要解释。/no_think"
        )
        output = f"{code_a}{code_b}{code_c}"
    elif variant == "a":
        instruction = "你是 itemic 分层编码器，只预测第一级 SID。"
        input_text = f"{prompt}\n\n物料域为 {begin}，请只输出 <s_a_...>。/no_think"
        output = code_a
    elif variant == "b":
        instruction = "你是 itemic 分层编码器，在已知第一级 SID 时预测第二级 SID。"
        input_text = f"{prompt}\n\n物料域为 {begin}，已知第一级为 {code_a}，请只输出 <s_b_...>。/no_think"
        output = code_b
    elif variant == "c":
        instruction = "你是 itemic 分层编码器，在已知前两级 SID 时预测第三级 SID。"
        input_text = (
            f"{prompt}\n\n物料域为 {begin}，已知前两级为 {code_a}{code_b}，请只输出 <s_c_...>。/no_think"
        )
        output = code_c
    else:
        raise ValueError(f"unsupported variant: {variant}")

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "history": [],
        "itemic": token,
        "variant": variant,
    }


def stable_key(messages: list[dict[str, str]]) -> str:
    token = target_itemic(messages)
    return hashlib.sha1(token.encode("utf-8")).hexdigest()

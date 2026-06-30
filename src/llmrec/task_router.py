from __future__ import annotations

import json
import re

from .data_schema import find_itemic_tokens, infer_domain_from_text


def infer_output_type(output: str) -> str:
    output = (output or "").strip()
    if not output:
        return "empty"
    try:
        parsed = json.loads(output)
        if isinstance(parsed, list):
            return "itemic_list" if all(find_itemic_tokens(str(x)) for x in parsed) else "json_list"
        if isinstance(parsed, dict):
            return "json"
    except Exception:
        pass
    tokens = find_itemic_tokens(output)
    if tokens and len(tokens) == 1 and output == tokens[0]:
        return "itemic_pattern"
    if tokens:
        return "itemic_list"
    if len(output) < 64 and re.fullmatch(r"[A-Da-d]|\d+", output):
        return "option"
    if any(word in output for word in ["因此", "因为", "首先", "推理", "分析"]):
        return "reasoning_answer"
    if len(output) < 80:
        return "short_answer"
    return "caption"


def infer_task_type(prompt: str, output: str = "") -> str:
    text = f"{prompt}\n{output}".lower()
    source_hints = {
        "懂推荐": "recommendation",
        "懂用户": "user_demand_understanding",
        "懂物料": "material_understanding",
        "懂世界": "world_knowledge",
    }
    for hint, task_type in source_hints.items():
        if hint in text:
            return task_type
    if any(k in text for k in ["next item", "下一", "接下来", "推荐", "目标内容", "top-k", "topk", "可能点击"]):
        return "recommendation"
    if any(k in text for k in ["懂用户", "用户", "user", "兴趣", "需求", "历史行为", "交互历史", "行为链", "denois", "噪声"]):
        return "user_demand_understanding"
    if find_itemic_tokens(text) and any(k in text for k in ["懂物料", "描述", "caption", "内容", "物料", "商品", "视频", "广告", "直播", "主播"]):
        return "material_understanding"
    if find_itemic_tokens(output):
        return "itemic_grounding"
    if any(k in text for k in ["常识", "知识", "数学", "代码", "cap 定理", "world"]):
        return "world_knowledge"
    if find_itemic_tokens(text):
        return "item_qa"
    return "other"


def infer_domain(prompt: str, output: str = "", fallback: str = "general") -> str:
    return infer_domain_from_text(f"{prompt}\n{output}", default=fallback)

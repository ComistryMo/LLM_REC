from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_schema import find_itemic_tokens
from .utils import read_jsonl, write_jsonl

STRONG_ACTIONS = ["purchase", "search", "follow", "like", "comment", "long_watch", "click", "short_watch", "exposure"]


def action_weight(action: str) -> int:
    action = (action or "").lower()
    for rank, name in enumerate(STRONG_ACTIONS):
        if name in action:
            return len(STRONG_ACTIONS) - rank
    return 0


def event_to_text(event: dict[str, Any]) -> str:
    parts = [
        f"[{event.get('timestamp', '')}]",
        f"[{event.get('action_type', 'unknown')}]",
        f"[{event.get('domain', 'general')}]",
    ]
    if event.get("query"):
        parts.append(f"query={event['query']}")
    if event.get("itemic_pattern"):
        parts.append(str(event["itemic_pattern"]))
    elif event.get("item_id"):
        parts.append(f"item_id={event['item_id']}")
    return " ".join(p for p in parts if p)


def history_prompt(events: list[dict[str, Any]], task: str) -> str:
    lines = "\n".join(event_to_text(e) for e in events)
    return f"用户历史行为：\n{lines}\n\n任务：{task}"


def pick_target(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    with_itemic = [e for e in events if e.get("itemic_pattern") and find_itemic_tokens(str(e.get("itemic_pattern")))]
    if not with_itemic:
        return None
    return max(with_itemic, key=lambda e: (action_weight(e.get("action_type", "")), e.get("timestamp", "")))


def build_next_item_sample(user_id: str, history: list[dict[str, Any]], target: dict[str, Any], sample_id: str) -> dict[str, Any]:
    domain = target.get("domain", "general")
    prompt = history_prompt(history, f"根据历史预测用户下一步最可能感兴趣的 {domain} itemic token。只输出 itemic token。")
    return {
        "id": sample_id,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(target["itemic_pattern"])}],
        "task_type": "recommendation",
        "output_type": "itemic_pattern",
        "domain": domain,
        "has_itemic": True,
        "has_cot": False,
        "input_len": len(prompt.split()),
        "output_len": 1,
        "itemic_count": 1,
        "source": "behavior_next_item",
        "user_id": user_id,
    }


def build_user_understanding_sample(user_id: str, history: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    strong = sorted(history, key=lambda e: action_weight(e.get("action_type", "")), reverse=True)[:5]
    domains = [e.get("domain", "general") for e in strong if e.get("domain")]
    interests = ", ".join(dict.fromkeys(domains)) or "general"
    answer = (
        f"用户近期强信号主要集中在 {interests}。"
        "搜索、购买、关注和长观看等行为应优先视为当前需求，短曝光或孤立点击可作为弱信号。"
    )
    prompt = history_prompt(history, "总结用户长期偏好、短期需求变化，并指出最关键的强反馈行为。")
    return {
        "id": sample_id,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        "task_type": "user_demand_understanding",
        "output_type": "reasoning_answer",
        "domain": "mixed" if len(set(domains)) > 1 else (domains[0] if domains else "general"),
        "has_itemic": bool(any(e.get("itemic_pattern") for e in history)),
        "has_cot": False,
        "input_len": len(prompt.split()),
        "output_len": len(answer.split()),
        "itemic_count": sum(len(find_itemic_tokens(str(e.get("itemic_pattern", "")))) for e in history),
        "source": "behavior_user_summary",
        "user_id": user_id,
    }


def build_history_sft(
    sequences_path: str | Path,
    output_user: str | Path,
    output_rec: str | Path,
    windows: list[int] | None = None,
    max_samples_per_user: int = 12,
) -> tuple[int, int]:
    windows = windows or [20, 50, 100]
    user_samples: list[dict[str, Any]] = []
    rec_samples: list[dict[str, Any]] = []
    for row in read_jsonl(sequences_path):
        user_id = str(row.get("user_id", ""))
        events = row.get("events") or []
        if len(events) < 2:
            continue
        made = 0
        split_start = max(1, int(len(events) * 0.8))
        candidate_indices = list(range(split_start, len(events)))
        for idx in candidate_indices:
            if made >= max_samples_per_user:
                break
            target = pick_target([events[idx]])
            if not target:
                continue
            for window in windows:
                start = max(0, idx - window)
                history = events[start:idx]
                if not history:
                    continue
                rec_samples.append(build_next_item_sample(user_id, history, target, f"rec-{user_id}-{idx}-{window}"))
                made += 1
                if made >= max_samples_per_user:
                    break
        if len(events) >= 3:
            history = events[-min(max(windows), len(events)) :]
            user_samples.append(build_user_understanding_sample(user_id, history, f"user-{user_id}-summary"))
    return write_jsonl(user_samples, output_user), write_jsonl(rec_samples, output_rec)


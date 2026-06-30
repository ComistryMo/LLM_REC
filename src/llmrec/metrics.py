from __future__ import annotations

import json
from typing import Iterable

from .data_schema import find_itemic_tokens, itemic_domains


def exact_match(pred: str, gold: str) -> bool:
    return (pred or "").strip() == (gold or "").strip()


def parse_candidates(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
        if isinstance(parsed, dict):
            for key in ["items", "itemic", "recommendations", "answer"]:
                if isinstance(parsed.get(key), list):
                    return [str(x).strip() for x in parsed[key]]
                if parsed.get(key):
                    return [str(parsed[key]).strip()]
    except Exception:
        pass
    tokens = find_itemic_tokens(text)
    return tokens or [text]


def pass_at_k(pred: str, gold: str, k: int = 5) -> bool:
    gold_set = set(parse_candidates(gold))
    return any(candidate in gold_set for candidate in parse_candidates(pred)[:k])


def itemic_valid_rate(predictions: Iterable[str]) -> float:
    values = list(predictions)
    if not values:
        return 0.0
    valid = sum(1 for pred in values if find_itemic_tokens(pred))
    return valid / len(values)


def domain_hit(pred: str, expected_domain: str) -> bool:
    domains = itemic_domains(pred)
    if expected_domain == "mixed":
        return bool(domains)
    return expected_domain in domains


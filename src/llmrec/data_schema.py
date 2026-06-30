from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

BEGIN_TAGS = {
    "video": "<|video_begin|>",
    "product": "<|prod_begin|>",
    "prod": "<|prod_begin|>",
    "ad": "<|ad_begin|>",
    "live": "<|living_begin|>",
    "living": "<|living_begin|>",
    "sid": "<|sid_begin|>",
}

BEGIN_TO_DOMAIN = {
    "<|video_begin|>": "video",
    "<|prod_begin|>": "product",
    "<|ad_begin|>": "ad",
    "<|living_begin|>": "live",
    "<|sid_begin|>": "general",
}

DOMAIN_VALUES = {"video", "product", "ad", "live", "mixed", "general"}

ITEMIC_RE = re.compile(
    r"(?P<begin><\|(?:video|prod|ad|living|sid)_begin\|>)"
    r"(?P<codes><s_a_[A-Za-z0-9_-]+><s_b_[A-Za-z0-9_-]+><s_c_[A-Za-z0-9_-]+>)"
)
LOOSE_CODE_RE = re.compile(r"<(?P<prefix>s_)?(?P<level>[abc])_(?P<value>[A-Za-z0-9_-]+)>")
BEGIN_RE = re.compile(r"<\|(?:video|prod|ad|living|live|sid)_(?:begin|end)\|>")
SECRET_RE = re.compile(
    r"(?i)("
    r"password\s*[:=]|passwd\s*[:=]|pwd\s*[:=]|cookie\s*[:=]|"
    r"token\s*[:=]\s*[\"']?(?!<\|)[A-Za-z0-9_.-]{16,}|bearer\s+[A-Za-z0-9_.-]{16,}|"
    r"secret[_-]?key\s*[:=]|access[_-]?key\s*[:=]|api[_-]?key\s*[:=]|"
    r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----|"
    r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}"
    r")"
)


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def normalize_itemic_text(text: str) -> str:
    """Normalize common draft/demo token variants to the official-ish form."""
    if not isinstance(text, str):
        return ""
    text = text.replace("<|live_begin|>", "<|living_begin|>")
    text = re.sub(r"<\|(?:video|prod|ad|living|live|sid)_end\|>", "", text)

    def code_repl(match: re.Match[str]) -> str:
        prefix = match.group("prefix") or ""
        level = match.group("level")
        value = match.group("value")
        if prefix == "s_":
            return match.group(0)
        return f"<s_{level}_{value}>"

    return LOOSE_CODE_RE.sub(code_repl, text)


def find_itemic_tokens(text: str, normalize: bool = True) -> list[str]:
    text = normalize_itemic_text(text) if normalize else (text or "")
    return [m.group(0) for m in ITEMIC_RE.finditer(text)]


def is_valid_itemic_token(token: str) -> bool:
    return bool(ITEMIC_RE.fullmatch(normalize_itemic_text(token).strip()))


def itemic_domains(text: str) -> set[str]:
    domains: set[str] = set()
    for match in ITEMIC_RE.finditer(normalize_itemic_text(text or "")):
        domains.add(BEGIN_TO_DOMAIN.get(match.group("begin"), "general"))
    return domains


def infer_domain_from_text(text: str, default: str = "general") -> str:
    domains = itemic_domains(text)
    if len(domains) == 1:
        return next(iter(domains))
    if len(domains) > 1:
        return "mixed"
    lowered = (text or "").lower()
    keyword_map = {
        "video": ["video", "短视频", "视频"],
        "product": ["product", "prod", "商品", "购买", "下单"],
        "ad": ["广告", "ad"],
        "live": ["直播", "主播", "live", "living"],
    }
    hits = {domain for domain, words in keyword_map.items() if any(w in lowered for w in words)}
    if len(hits) == 1:
        return next(iter(hits))
    if len(hits) > 1:
        return "mixed"
    return default


def message_text(messages: Iterable[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def token_len(text: str) -> int:
    return len((text or "").split())

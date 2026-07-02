#!/usr/bin/env python3
"""parquet → Alpaca JSONL converter v3 with filter logging.

Adds --filter-log argument: every dropped/transformed/skipped sample is logged
to a JSONL file with full content + reason for downstream analysis.

Filter reasons logged:
- drop:itemic_overflow      itemic token alphabet exceeds --max_token_types
- skip:no_messages          parquet row has no messages
- skip:exception            messages JSON parse / convert exception
- skip:to_alpaca_empty      user_messages or assistant_messages empty
- transform:filter_sid      tokens were deleted/normalized (kept but logged optionally)
- transform:think_inject    think pattern was injected (kept but logged optionally)
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd


_TOKENS_TO_DELETE = [
    "<|sid_end|>",
    "<|goods_sid_end|>",
    "<|living_end|>",
    "<|ad_end|>",
    "<|prod_end|>",
    "<|video_end|>",
]

_TOKENS_TO_NORMALIZE = [
    ("<|live_begin|>", "<|living_begin|>"),
    ("<prod_s_", "<s_"),
    ("<|pid_video_begin|>",  "<pid_video_begin>"),
    ("<|pid_video_end|>",    "<pid_video_end>"),
    ("<|pid_ad_begin|>",     "<pid_ad_begin>"),
    ("<|pid_ad_end|>",       "<pid_ad_end>"),
    ("<|pid_prod_begin|>",   "<pid_prod_begin>"),
    ("<|pid_prod_end|>",     "<pid_prod_end>"),
    ("<|pid_living_begin|>", "<pid_living_begin>"),
    ("<|pid_living_end|>",   "<pid_living_end>"),
]

_ITEMIC_TOKEN_RE = re.compile(r"<s_([a-z])_\d+>")


def filter_sid_end_tokens(text: str, stats: dict | None = None,
                          token_hits: dict | None = None) -> str:
    for tok in _TOKENS_TO_DELETE:
        if tok in text:
            cnt = text.count(tok)
            if stats is not None:
                stats[f"delete:{tok}"] += cnt
            if token_hits is not None:
                token_hits[f"delete:{tok}"] = token_hits.get(f"delete:{tok}", 0) + cnt
            text = text.replace(tok, "")
    for src, dst in _TOKENS_TO_NORMALIZE:
        if src in text:
            cnt = text.count(src)
            if stats is not None:
                stats[f"normalize:{src}"] += cnt
            if token_hits is not None:
                token_hits[f"normalize:{src}"] = token_hits.get(f"normalize:{src}", 0) + cnt
            text = text.replace(src, dst)
    return text


def check_itemic_token_types(text: str, max_token_types: int):
    found = set(_ITEMIC_TOKEN_RE.findall(text))
    return len(found) <= max_token_types, found


def convert_messages(messages: list, add_think_pattern: bool,
                     do_filter_sid: bool, stats: dict | None,
                     row_token_hits: dict | None,
                     row_think_events: list | None) -> list:
    msg_list = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            text = content
        elif isinstance(content, dict) and content.get("type") == "text":
            text = content["text"]
        elif isinstance(content, list):
            text = "".join(
                c["text"] if isinstance(c, dict) and c.get("type") == "text" else c
                for c in content
                if isinstance(c, (str, dict))
            )
        else:
            raise ValueError(f"Unsupported content type: {type(content)}, value={content!r}")

        if do_filter_sid:
            text = filter_sid_end_tokens(text, stats, row_token_hits)

        msg_list.append({"role": role, "content": text})

    if add_think_pattern:
        for i, msg in enumerate(msg_list):
            if msg["role"] != "assistant":
                continue
            user_idx = i - 1
            if user_idx < 0 or msg_list[user_idx]["role"] != "user":
                continue

            match = re.search(r"<think>(.*?)</think>", msg["content"], re.DOTALL)
            if match is None:
                msg_list[user_idx]["content"] += "/no_think"
                msg_list[i]["content"] = "<think>\n\n</think>\n" + msg["content"]
                if stats is not None:
                    stats["think:inject_empty"] += 1
                if row_think_events is not None:
                    row_think_events.append("inject_empty")
            elif match.group(1).strip():
                msg_list[user_idx]["content"] += "/think"
                if stats is not None:
                    stats["think:keep_existing"] += 1
                if row_think_events is not None:
                    row_think_events.append("keep_existing")
            else:
                msg_list[user_idx]["content"] += "/no_think"
                if stats is not None:
                    stats["think:empty_tag"] += 1
                if row_think_events is not None:
                    row_think_events.append("empty_tag")

    return msg_list


def to_alpaca(msg_list: list):
    instruction = ""
    for msg in msg_list:
        if msg["role"] == "system":
            instruction = msg["content"]
            break

    user_messages = []
    assistant_messages = []

    for msg in msg_list:
        if msg["role"] in ("user", "human"):
            user_messages.append(msg["content"])
        elif msg["role"] == "assistant":
            assistant_messages.append(msg["content"])

    if not user_messages or not assistant_messages:
        return None

    input_text = user_messages[0]
    output_text = assistant_messages[-1]

    record = {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "history": [],
    }

    if len(user_messages) > 1 or len(assistant_messages) > 1:
        num_history_pairs = min(len(user_messages) - 1, len(assistant_messages))
        for i in range(num_history_pairs):
            record["history"].append([user_messages[i], assistant_messages[i]])

    return record


def _messages_preview(raw_messages, char_limit=4000):
    """Stringify messages safely for logging."""
    if isinstance(raw_messages, str):
        s = raw_messages
    else:
        try:
            s = json.dumps(raw_messages, ensure_ascii=False)
        except Exception:
            s = repr(raw_messages)
    if len(s) > char_limit:
        return s[:char_limit] + f"...[truncated, full_len={len(s)}]"
    return s


def process_parquet(path: str, args, stats: dict, filter_logger):
    df = pd.read_parquet(path)
    records = []
    skipped = 0
    dropped_itemic = 0

    for row_idx, row in df.iterrows():
        raw = row.get("messages")
        row_uuid = row.get("uuid")
        row_source = row.get("source")
        row_line_id = row.get("line_id")
        row_base = {
            "file": str(path),
            "row_idx": int(row_idx) if hasattr(row_idx, "__int__") else row_idx,
            "uuid": row_uuid if isinstance(row_uuid, str) else (str(row_uuid) if row_uuid is not None else None),
            "source": row_source if isinstance(row_source, str) else (str(row_source) if row_source is not None else None),
            "line_id": row_line_id if isinstance(row_line_id, str) else (str(row_line_id) if row_line_id is not None else None),
        }

        if raw is None or isinstance(raw, float):
            skipped += 1
            stats["skip:no_messages"] += 1
            if filter_logger is not None:
                filter_logger({
                    **row_base,
                    "reason": "skip:no_messages",
                    "raw_messages": None,
                })
            continue

        try:
            messages = json.loads(raw) if isinstance(raw, str) else raw
            row_token_hits = {}
            row_think_events = []
            msg_list = convert_messages(
                messages,
                add_think_pattern=args.add_think_pattern,
                do_filter_sid=args.filter_sid_tokens,
                stats=stats,
                row_token_hits=row_token_hits,
                row_think_events=row_think_events,
            )

            if args.max_token_types is not None:
                full_text = "".join(m["content"] for m in msg_list)
                ok, found = check_itemic_token_types(full_text, args.max_token_types)
                if not ok:
                    dropped_itemic += 1
                    stats["dropped:itemic_overflow"] += 1
                    found_key = ",".join(sorted(found))
                    stats[f"itemic_set:{found_key}"] += 1
                    if filter_logger is not None:
                        filter_logger({
                            **row_base,
                            "reason": "drop:itemic_overflow",
                            "max_token_types": args.max_token_types,
                            "itemic_letters_found": sorted(found),
                            "token_hits": row_token_hits,
                            "messages_after_convert": msg_list,
                            "raw_messages_preview": _messages_preview(raw),
                        })
                    continue

            record = to_alpaca(msg_list)
            if record is None:
                stats["skip:to_alpaca_empty"] += 1
                skipped += 1
                if filter_logger is not None:
                    filter_logger({
                        **row_base,
                        "reason": "skip:to_alpaca_empty",
                        "token_hits": row_token_hits,
                        "messages_after_convert": msg_list,
                        "raw_messages_preview": _messages_preview(raw),
                    })
                continue
            records.append(record)

            if args.log_kept_transforms and filter_logger is not None and (row_token_hits or row_think_events):
                filter_logger({
                    **row_base,
                    "reason": "kept:transform",
                    "token_hits": row_token_hits,
                    "think_events": row_think_events,
                })

        except Exception as e:
            stats["skip:exception"] += 1
            skipped += 1
            if filter_logger is not None:
                filter_logger({
                    **row_base,
                    "reason": "skip:exception",
                    "error": repr(e),
                    "raw_messages_preview": _messages_preview(raw),
                })
            print(f"[WARN] skipping row due to: {e}", file=sys.stderr)

    print(
        f"[INFO] {path}: {len(records)} converted, {skipped} skipped, {dropped_itemic} dropped(itemic)",
        file=sys.stderr,
    )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True,
                        help="parquet 文件/目录/glob")
    parser.add_argument("--output", required=True, help="输出 jsonl 路径")
    parser.add_argument("--max_token_types", type=int, default=3,
                        help="允许 <s_X_> 字母种类数(默认 3 = a/b/c)。设 None 关闭检查")
    parser.add_argument("--no_filter_sid_tokens", dest="filter_sid_tokens", action="store_false")
    parser.add_argument("--no_add_think_pattern", dest="add_think_pattern", action="store_false")
    parser.add_argument("--report", action="store_true", help="打印变换统计")
    parser.add_argument("--filter-log", default=None,
                        help="过滤/转换日志输出路径 (JSONL)。每条 dropped/skipped/transformed 样本一行")
    parser.add_argument("--log-kept-transforms", action="store_true",
                        help="日志中也记录保留但发生过 token 替换或 think 注入的样本")
    parser.add_argument("--summary", default=None,
                        help="可选：把最终 stats summary 也写到一个 JSON 文件")
    parser.add_argument("--shuffle", action="store_true",
                        help="对最终记录全局随机打乱后再写出（仍保留首条有 history 的记录在最前）")
    parser.add_argument("--shuffle-seed", type=int, default=2026, help="shuffle 的随机种子")
    parser.set_defaults(filter_sid_tokens=True, add_think_pattern=True)
    args = parser.parse_args()

    stats = collections.Counter()
    filter_log_fp = None
    if args.filter_log:
        Path(args.filter_log).parent.mkdir(parents=True, exist_ok=True)
        filter_log_fp = open(args.filter_log, "w", encoding="utf-8")
    filter_log_count = [0]

    def filter_logger(payload):
        if filter_log_fp is None:
            return
        filter_log_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        filter_log_count[0] += 1

    all_records = []
    for pattern in args.input:
        if "*" in pattern:
            from glob import glob
            paths = sorted(glob(pattern, recursive=True))
            paths = [Path(p) for p in paths]
        else:
            p = Path(pattern)
            if p.is_dir():
                paths = sorted(p.rglob("*.parquet"))
            else:
                paths = [p]
        for p in paths:
            all_records.extend(process_parquet(str(p), args, stats, filter_logger))

    # Optional shuffle
    if args.shuffle:
        import random
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(all_records)
        print(f"[INFO] shuffled {len(all_records)} records (seed={args.shuffle_seed})", file=sys.stderr)

    # move first record with history to front (datasets type inference)
    first_hist_idx = next((i for i, r in enumerate(all_records) if r and r.get("history")), None)
    if first_hist_idx is not None and first_hist_idx > 0:
        all_records.insert(0, all_records.pop(first_hist_idx))
        print(
            f"[INFO] moved record {first_hist_idx} to front (has history, avoids datasets null-type inference)",
            file=sys.stderr,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[OK] Written {len(all_records)} samples to {args.output}", file=sys.stderr)

    if filter_log_fp is not None:
        filter_log_fp.close()
        print(f"[OK] Filter log written: {filter_log_count[0]} entries -> {args.filter_log}",
              file=sys.stderr)

    summary_payload = {
        "input": args.input,
        "output": args.output,
        "records_written": len(all_records),
        "filter_log": args.filter_log,
        "filter_log_entries": filter_log_count[0],
        "stats": dict(stats),
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.report:
        print("\n=== 统计报告 ===", file=sys.stderr)
        print(f"\n[token filter 命中次数]", file=sys.stderr)
        for k in sorted(stats.keys()):
            if k.startswith(("delete:", "normalize:")):
                print(f"  {k:<45} {stats[k]:>10,}", file=sys.stderr)
        print(f"\n[think pattern 注入次数]", file=sys.stderr)
        for k in sorted(stats.keys()):
            if k.startswith("think:"):
                print(f"  {k:<45} {stats[k]:>10,}", file=sys.stderr)
        print(f"\n[skip 原因]", file=sys.stderr)
        for k in sorted(stats.keys()):
            if k.startswith("skip:"):
                print(f"  {k:<45} {stats[k]:>10,}", file=sys.stderr)
        print(f"\n[itemic 字母种类超限丢弃]", file=sys.stderr)
        print(
            f"  dropped:itemic_overflow             {stats.get('dropped:itemic_overflow', 0):>10,}",
            file=sys.stderr,
        )
        top_sets = [(k.split(":", 1)[1], v) for k, v in stats.items() if k.startswith("itemic_set:")]
        if top_sets:
            print(f"\n  丢弃样本的 itemic 字母组合(top 10):", file=sys.stderr)
            for s, n in sorted(top_sets, key=lambda x: -x[1])[:10]:
                print(f"    {{ {s} }}  →  {n:,} 条", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ITEMIC_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|>"
    r"<s_a_\d+><s_b_\d+><s_c_\d+>"
)


def locate_main_file(directory: Path) -> Path:
    files = list(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {directory}")
    return max(files, key=lambda path: path.stat().st_size)


def unwrap_record(line: str, line_number: int) -> dict[str, str]:
    value = json.loads(line)
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], dict):
            raise ValueError(f"line {line_number}: expected a one-record list")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: expected an object")
    result = {key: str(value.get(key, "")) for key in ("system", "prompt", "response")}
    if not result["prompt"] or not result["response"]:
        raise ValueError(f"line {line_number}: empty prompt or response")
    return result


def item_key(record: dict[str, str]) -> str:
    text = "\n".join(record.values())
    match = ITEMIC_RE.search(text)
    if match:
        return match.group(0)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def is_validation(key: str, buckets: int, validation_buckets: int) -> bool:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16) % buckets
    return bucket < validation_buckets


def convert_stage(
    source: Path,
    train_path: Path,
    valid_path: Path,
    stage: str,
    buckets: int,
    validation_buckets: int,
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8") as input_stream,
        train_path.open("w", encoding="utf-8") as train_stream,
        valid_path.open("w", encoding="utf-8") as valid_stream,
    ):
        for line_number, line in enumerate(input_stream, 1):
            record = unwrap_record(line, line_number)
            token_match = ITEMIC_RE.search("\n".join(record.values()))
            if token_match is None:
                raise ValueError(f"{source}:{line_number}: no valid itemic token")
            if stage == "ep1":
                has_think_control = "/think" in record["prompt"]
                has_no_think_control = "/no_think" in record["prompt"]
                if (
                    has_think_control == has_no_think_control
                    or "<think>" not in record["response"]
                    or "</think>" not in record["response"]
                ):
                    raise ValueError(f"{source}:{line_number}: invalid EP1 control/think sample")
                counts["control:think" if has_think_control else "control:no_think_official"] += 1
            else:
                if "/no_think" not in record["prompt"] or "<think>" in record["response"]:
                    raise ValueError(f"{source}:{line_number}: invalid EP2 no-think sample")
                if ITEMIC_RE.fullmatch(record["response"].strip()) is None:
                    raise ValueError(f"{source}:{line_number}: EP2 response is not itemic-only")

            messages = []
            if record["system"]:
                messages.append({"role": "system", "content": record["system"]})
            messages.extend(
                [
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": record["response"]},
                ]
            )
            output = {"messages": messages}
            key = item_key(record)
            validation = is_validation(key, buckets, validation_buckets)
            stream = valid_stream if validation else train_stream
            stream.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["valid" if validation else "train"] += 1
            domain = token_match.group(0).split("_begin", 1)[0].split("|")[-1]
            counts[f"domain:{domain}"] += 1

    return {
        "source": str(source),
        "train": str(train_path),
        "valid": str(valid_path),
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe Swift datasets for material EP1/EP2.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/hz/onereason_competition/data/material_2ep"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/hz/onereason_competition/data/material_2ep/swift_messages_v2"),
    )
    parser.add_argument("--buckets", type=int, default=1000)
    parser.add_argument("--validation-buckets", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.validation_buckets < args.buckets:
        raise ValueError("validation-buckets must be between zero and buckets")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_root}")

    ep1_source = locate_main_file(args.source_root / "ep1_think")
    ep2_source = locate_main_file(args.source_root / "ep2_no_think")
    reports = {
        "ep1": convert_stage(
            ep1_source,
            args.output_root / "ep1_train.jsonl",
            args.output_root / "ep1_valid.jsonl",
            "ep1",
            args.buckets,
            args.validation_buckets,
        ),
        "ep2": convert_stage(
            ep2_source,
            args.output_root / "ep2_train.jsonl",
            args.output_root / "ep2_valid.jsonl",
            "ep2",
            args.buckets,
            args.validation_buckets,
        ),
    }
    report_path = args.output_root / "PREPARE_REPORT.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from itemic_teacher_forced import evaluate_loaded_model, prepare_examples


def find_weights(path: str) -> Path:
    root = Path(path)
    direct = root / "model.safetensors"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"single-file safetensors weights not found under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate linear interpolation between base and tuned full models.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--alpha", type=float, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-alpha", type=float)
    parser.add_argument("--save-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples, data_stats = prepare_examples(args.validation, tokenizer, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.tuned, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()

    results = {}
    base_path = find_weights(args.base)
    tuned_path = find_weights(args.tuned)
    with safe_open(base_path, framework="pt", device="cpu") as base_weights, safe_open(
        tuned_path, framework="pt", device="cpu"
    ) as tuned_weights:
        available = set(base_weights.keys()) & set(tuned_weights.keys())
        missing = [name for name, _ in model.named_parameters() if name not in available]
        if missing:
            raise SystemExit(f"weights missing from safetensors: {missing[:5]}")
        for alpha in sorted(set(args.alpha)):
            if not 0 <= alpha <= 1:
                raise SystemExit(f"alpha must be in [0,1]: {alpha}")
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    base_tensor = base_weights.get_tensor(name)
                    tuned_tensor = tuned_weights.get_tensor(name)
                    merged = base_tensor.float().lerp(tuned_tensor.float(), alpha)
                    parameter.copy_(merged.to(device=parameter.device, dtype=parameter.dtype))
            name = f"alpha_{alpha:g}"
            results[name] = evaluate_loaded_model(
                model, f"base+{alpha:g}*(tuned-base)", examples, tokenizer, args.device, args.batch_size
            )
            print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
            if args.save_alpha is not None and abs(alpha - args.save_alpha) < 1e-12:
                if args.save_directory is None:
                    raise SystemExit("--save-directory is required with --save-alpha")
                model.save_pretrained(args.save_directory, safe_serialization=True)
                tokenizer.save_pretrained(args.save_directory)

    payload = {
        "base": args.base,
        "tuned": args.tuned,
        "data": data_stats,
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

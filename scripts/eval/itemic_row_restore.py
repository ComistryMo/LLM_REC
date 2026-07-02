#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from itemic_teacher_forced import evaluate_loaded_model, prepare_examples


ITEMIC_VOCAB_RE = re.compile(
    r"(?:<\|(?:video|prod|ad|living|sid)_begin\|>|<s_[abc]_[A-Za-z0-9_-]+>)"
)


def copy_rows(weight: torch.Tensor, ids: torch.Tensor, rows: torch.Tensor) -> None:
    weight.index_copy_(0, ids.to(weight.device), rows.to(device=weight.device, dtype=weight.dtype))


def blend(base: torch.Tensor, tuned: torch.Tensor, base_ratio: float) -> torch.Tensor:
    return tuned.float().lerp(base.float(), base_ratio).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate restoring itemic embedding/lm_head rows from the base model.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-variant", choices=["lm_head", "embed", "both", "blend50"])
    parser.add_argument("--save-path", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ids = sorted(token_id for token, token_id in tokenizer.get_vocab().items() if ITEMIC_VOCAB_RE.fullmatch(token))
    if not ids:
        raise SystemExit("no itemic vocabulary rows found")
    index = torch.tensor(ids, dtype=torch.long)
    examples, data_stats = prepare_examples(args.validation, tokenizer, args.max_length)

    tuned = AutoModelForCausalLM.from_pretrained(
        args.tuned, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    base = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        tuned_embed = tuned.get_input_embeddings().weight.index_select(0, index.to(args.device)).cpu()
        tuned_head = tuned.get_output_embeddings().weight.index_select(0, index.to(args.device)).cpu()
        base_embed = base.get_input_embeddings().weight.index_select(0, index).cpu()
        base_head = base.get_output_embeddings().weight.index_select(0, index).cpu()
    del base
    gc.collect()

    variants = {
        "tuned": (tuned_embed, tuned_head),
        "lm_head": (tuned_embed, base_head),
        "embed": (base_embed, tuned_head),
        "both": (base_embed, base_head),
        "blend50": (blend(base_embed, tuned_embed, 0.5), blend(base_head, tuned_head, 0.5)),
    }
    results = {}
    with torch.no_grad():
        for name, (embed_rows, head_rows) in variants.items():
            copy_rows(tuned.get_input_embeddings().weight, index, embed_rows)
            copy_rows(tuned.get_output_embeddings().weight, index, head_rows)
            results[name] = evaluate_loaded_model(
                tuned, f"{args.tuned}:{name}", examples, tokenizer, args.device, args.batch_size
            )
            print(name, json.dumps(results[name], ensure_ascii=False), flush=True)

        if args.save_variant:
            embed_rows, head_rows = variants[args.save_variant]
            copy_rows(tuned.get_input_embeddings().weight, index, embed_rows)
            copy_rows(tuned.get_output_embeddings().weight, index, head_rows)
            if args.save_path is None:
                raise SystemExit("--save-path is required with --save-variant")
            args.save_path.mkdir(parents=True, exist_ok=True)
            tuned.save_pretrained(args.save_path, safe_serialization=True)
            tokenizer.save_pretrained(args.save_path)

    payload = {
        "base": args.base,
        "tuned": args.tuned,
        "itemic_vocab_rows": len(ids),
        "data": data_stats,
        "variants": results,
        "saved_variant": args.save_variant,
        "save_path": str(args.save_path) if args.save_path else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

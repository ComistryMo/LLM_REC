#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ITEMIC_PROBES = [
    "<|video_begin|><s_a_1234><s_b_5678><s_c_7012>",
    "<|prod_begin|><s_a_101><s_b_202><s_c_303>",
    "<|ad_begin|><s_a_11><s_b_22><s_c_33>",
    "<|living_begin|><s_a_1><s_b_2><s_c_3>",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test tokenizer/model loading.")
    parser.add_argument("--model", default="/data/hz/models/OneReason-0.8B-pretrain-competition")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device_map,
    )
    probe_tokens = {text: tokenizer.convert_ids_to_tokens(tokenizer.encode(text, add_special_tokens=False)) for text in ITEMIC_PROBES}
    report = {
        "model": args.model,
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "cuda_available": torch.cuda.is_available(),
        "itemic_probe_tokens": probe_tokens,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

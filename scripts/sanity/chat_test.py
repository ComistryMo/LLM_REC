#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "请简要描述物料 <|video_begin|><s_a_1234><s_b_5678><s_c_9012> 可能对应的内容类型。",
    "用户历史行为：[watch][video]<|video_begin|><s_a_12><s_b_34><s_c_56>\\n[search] 露营装备\\n请只输出一个可能感兴趣的商品 itemic token。",
    "CAP 定理是什么意思？",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small generation sanity checks.")
    parser.add_argument("--model", default="/data/hz/models/OneReason-0.8B-pretrain-competition")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False)
        print("=" * 80)
        print(prompt)
        print("-" * 80)
        print(decoded.strip())


if __name__ == "__main__":
    main()


# LLaMA-Factory training

LLaMA-Factory uses `configs/llamafactory/lora_sft.yaml`.

Before training, run `scripts/data/mix_datasets.py`; it writes ShareGPT-compatible JSONL files under `data/processed/`.

```bash
bash scripts/train/train_llamafactory_lora.sh
```


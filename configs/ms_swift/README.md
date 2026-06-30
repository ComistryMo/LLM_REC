# ms-swift configs

The shell entrypoints in `scripts/train/` build Swift arguments dynamically because Swift CLI option names vary by version.

Default paths:

- model: `/data/hz/models/OneReason-0.8B-pretrain-competition`
- train data: `data/processed/train_mix_v1.jsonl`
- valid data: `data/processed/valid_mix_v1.jsonl`
- LoRA output: `outputs/swift_lora_v1`
- full SFT output: `outputs/swift_full_v1`

Override with environment variables such as `MODEL_PATH`, `TRAIN_DATA`, `MAX_LENGTH`, `LR`, `EPOCHS`, `BATCH_SIZE`, and `GRAD_ACCUM`.


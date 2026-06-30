# Server Configuration Audit

Audit date: 2026-06-30

Remote host: `10.15.10.64`

Project path: `/data/hz/llmrec_competition`

Model path: `/data/hz/models/OneReason-0.8B-pretrain-competition`

## Sources Checked

- Competition official site: https://ks-llmrec.streamlake.com/
- Wanqing competition guide: https://www.streamlake.com/document/WANQING/mq57afym1d7p20atnau
- Wanqing dev-machine guide: https://www.streamlake.com/document/WANQING/mh1g8b8aunh8esspfm
- Hugging Face model card: https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition
- OneReason technical report: https://arxiv.org/abs/2606.06260
- Public baseline/reference post inspected: https://github.com/Lsw-1220/llmrec

## Server Snapshot

| Item | Server value | Requirement / expectation | Status |
|---|---|---|---|
| Repository | `/data/hz/llmrec_competition` | Project initialized under `/data/hz` without deleting existing files | OK |
| Base model | `/data/hz/models/OneReason-0.8B-pretrain-competition` | Official model is `OneReason-0.8B-pretrain-competition` | OK |
| Model load | `Qwen3ForCausalLM`, `Qwen2Tokenizer`, chat template present | Model card supports `AutoTokenizer` / `AutoModelForCausalLM` and chat template usage | OK |
| GPU | `6 x NVIDIA A100-SXM4-80GB` | A100 available for LoRA/full SFT | OK, currently busy |
| Driver / CUDA | Driver `570.169`, system CUDA `12.8` | CUDA GPU runtime available | OK |
| Conda env | `onereason-rec` | Isolated training environment | OK |
| Python | `3.11.15` | Python >= 3.10 | OK |
| PyTorch | `2.5.1+cu121` | PyTorch installed with CUDA | OK |
| Transformers | `5.3.0` | Wanqing FAQ recommends offline training with Transformers `v5.3.0` | OK |
| ms-swift | `4.3.2` | Wanqing dev-machine path mentions PyTorch and ms-swift images | OK |
| LLaMA-Factory | `0.9.5` | Optional backup training entry | OK |
| datasets / accelerate / peft / trl | Installed | Required for SFT/LoRA pipeline | OK |
| numpy | `1.26.4` | Needed by torch/transformers stack | OK, pinned for old virtual CPU |
| polars | `1.33.1` via `polars-lts-cpu` | Data processing dependency | OK with warning |
| Model files | `model.safetensors`, tokenizer/config files present | Full base checkpoint available locally | OK |
| GitHub sync | `origin/main` pushed to `ComistryMo/LLM_REC` | Source code backed up remotely | OK |

## Compliance Notes

- Wanqing states the competition model supports text generation and SFT; the repo uses `messages` SFT format and trains via ms-swift/LLaMA-Factory SFT entrypoints.
- Wanqing supports LoRA and full update; both `train_swift_lora.sh` and `train_swift_full.sh` are present, plus a LLaMA-Factory LoRA fallback.
- Wanqing manual upload requirements are covered by export/pre-submit scripts: LoRA checks for adapter files, full model checks for model safetensors/index.
- Official evaluation reports four dimensions: material, user, recommendation, and world knowledge. The repo implements matching local EDA/eval buckets.
- Official FAQ allows external/general/custom recommendation data but requires reproducible data and scripts during reproduction. The repo defaults to official data first and keeps generated data out of git.
- Official FAQ says model structure and config parameters are checked against the baseline. The repo downloads the official base model and does not alter model architecture.
- OneReason itemic token format is domain begin token plus `<s_a_*>`, `<s_b_*>`, `<s_c_*>`; the repo validates and normalizes this format.

## Known Warnings

- The remote virtual CPU is reported as old Core2-class and lacks some modern CPU flags. `numpy==1.26.4` and `polars-lts-cpu==1.33.1` are pinned to avoid incompatible wheels.
- `polars-lts-cpu` still emits a CPU feature warning for `sse4.1` and `popcnt`, but basic Polars operations passed. For large EDA, pandas/pyarrow remain the safer fallback if Polars crashes.
- Current GPU and system memory are heavily occupied. Environment checks and model sanity pass, but full training should wait for free GPU memory.
- The official website is partially dynamic; Wanqing docs and the model card provide the concrete operational requirements used for this audit.

## Verified Commands

```bash
cd /data/hz/llmrec_competition
conda activate onereason-rec
python scripts/sanity/load_model.py --model /data/hz/models/OneReason-0.8B-pretrain-competition --device-map auto
python scripts/data/parse_sft.py data/samples/sample_sft.jsonl
python scripts/data/parse_behavior.py data/samples/sample_behavior.jsonl
python scripts/data/build_history_sft.py
python scripts/data/mix_datasets.py
python scripts/data/check_format.py data/processed/train_mix_v1.jsonl data/processed/valid_mix_v1.jsonl
python scripts/eda/eda_report.py
python scripts/eval/local_eval.py
python -m pytest -q
```

Result: model load and generation sanity passed; sample data pipeline passed; `pytest` passed with `7 passed`.


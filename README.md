# 快手探索者 LLM-Rec 挑战赛工程仓库

本仓库用于基于 `OpenOneRec/OneReason-0.8B-pretrain-competition` 快速完成 EDA、数据构造、SFT 训练、模型导出和提交前检查。

默认远程路径：

```text
/data/hz/llmrec_competition
/data/hz/models/OneReason-0.8B-pretrain-competition
```

## 目录结构

```text
docs/                 规则、路线、环境、runbook
data/                 raw/interim/processed/eda/samples
configs/              ms-swift、LLaMA-Factory、DeepSpeed 配置
scripts/              setup/data/eda/train/eval/export/sanity 入口
src/llmrec/           可复用数据、校验、指标代码
training/             框架入口说明
outputs/              训练和导出产物，默认不入 git
logs/                 环境检查和训练日志，默认不入 git
tests/                pytest 单元测试
```

## 快速开始

```bash
cd /data/hz/llmrec_competition
bash scripts/setup/check_env.sh
bash scripts/setup/create_conda_env.sh
conda activate onereason-rec
bash scripts/setup/download_model.sh
python scripts/sanity/load_model.py --model /data/hz/models/OneReason-0.8B-pretrain-competition
```

如果 HuggingFace 访问慢，可在运行下载脚本前手动设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

脚本不会写死镜像。

## 数据放置

- 官方 SFT 数据放入 `data/raw/`。
- 用户行为数据放入 `data/raw/`。
- 小样例保留在 `data/samples/`，可用于验证链路。

正式字段开放后，优先修改 `configs/behavior_schema_aliases.yaml`，不要直接改解析代码。

## 数据处理

```bash
python scripts/data/extract_dataset.py data/raw/dataset.tar.gz --output data/raw/official
python scripts/data/parse_sft.py data/raw/official/*.jsonl \
  --tokenizer /data/hz/models/OneReason-0.8B-pretrain-competition
python scripts/data/split_official.py
python scripts/data/check_format.py \
  data/processed/train_official_v1.jsonl data/processed/valid_official_v1.jsonl \
  --max-length 8192

# 正式行为明细开放后再运行以下构造链路
python scripts/data/parse_behavior.py data/raw/user_behavior.jsonl
python scripts/data/build_history_sft.py
python scripts/data/mix_datasets.py
```

## EDA

```bash
python scripts/eda/eda_report.py
cat data/eda/EDA_REPORT.md
```

报告包含样本量、任务/输出/domain/action 分布、用户序列长度 P50/P90/P99、输入输出长度、itemic token 合法率、热门 token 和异常样本统计。本批官方数据结论见 `docs/OFFICIAL_DATA_EDA.md`。

## 训练

首选 ms-swift LoRA baseline：

```bash
bash scripts/train/train_swift_lora.sh
```

ms-swift 全参 SFT：

```bash
DEEPSPEED=/data/hz/llmrec_competition/configs/deepspeed/zero2.json \
bash scripts/train/train_swift_full.sh
```

LLaMA-Factory LoRA 备选：

```bash
bash scripts/train/train_llamafactory_lora.sh
```

常用覆盖参数：

```bash
MAX_LENGTH=8192 LR=1e-4 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=16 bash scripts/train/train_swift_lora.sh
```

## 验证与导出

```bash
python scripts/eval/local_eval.py
python scripts/sanity/chat_test.py --model outputs/swift_lora_v1
bash scripts/export/export_swift_lora.sh
python scripts/export/pre_submit_check.py --model-dir outputs/export_swift_lora_v1
```

## 注意事项

- 不提交模型权重、原始数据、日志、token、cookie、密码或 `.env`。
- LoRA 训练优先覆盖 attention 和 MLP 常见线性层；如框架支持，关注 `embed_tokens` 和 `lm_head` 是否保存或可训练。
- SFT 训练默认只学习 assistant 输出，避免模型复述长用户历史。
- 第一版 baseline 只使用官方 SFT 数据；用户行为构造样本作为第二阶段增量。
- 当前没有免密 SSH 时，不要把密码写入命令行或脚本。请使用 SSH key、平台终端或手动同步仓库。

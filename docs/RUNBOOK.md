# Runbook

## 从零到训练

1. 登录机器

   ```bash
   ssh root@10.15.10.64
   cd /data/hz/llmrec_competition
   ```

2. 激活环境

   ```bash
   conda activate onereason-rec
   ```

3. 检查 GPU、磁盘、内存

   ```bash
   bash scripts/setup/check_env.sh
   ```

4. 下载模型

   ```bash
   bash scripts/setup/download_model.sh
   python scripts/sanity/load_model.py --model /data/hz/models/OneReason-0.8B-pretrain-competition
   ```

5. 放置数据

   ```bash
   ls -lh data/raw/
   ```

6. 运行 EDA 和格式解析

   ```bash
   python scripts/data/parse_sft.py data/raw/official_sft.jsonl
   python scripts/data/parse_behavior.py data/raw/user_behavior.jsonl
   python scripts/eda/eda_report.py
   ```

7. 构造训练集

   ```bash
   python scripts/data/build_history_sft.py
   python scripts/data/mix_datasets.py
   python scripts/data/check_format.py data/processed/train_mix_v1.jsonl data/processed/valid_mix_v1.jsonl
   ```

8. 跑 LoRA baseline

   ```bash
   bash scripts/train/train_swift_lora.sh
   ```

9. 跑本地验证

   ```bash
   python scripts/eval/local_eval.py
   python scripts/sanity/chat_test.py --model outputs/swift_lora_v1
   ```

10. 导出模型

   ```bash
   bash scripts/export/export_swift_lora.sh
   ```

11. 提交前检查

   ```bash
   python scripts/export/pre_submit_check.py --model-dir outputs/export_swift_lora_v1
   find outputs/export_swift_lora_v1 -maxdepth 2 -type f -printf "%p %s\n"
   ```

## 样例链路

没有官方数据时，可以跑：

```bash
python scripts/data/parse_sft.py data/samples/sample_sft.jsonl
python scripts/data/parse_behavior.py data/samples/sample_behavior.jsonl
python scripts/data/build_history_sft.py
python scripts/data/mix_datasets.py
python scripts/data/check_format.py data/processed/train_mix_v1.jsonl data/processed/valid_mix_v1.jsonl
python scripts/eda/eda_report.py
python scripts/eval/local_eval.py
```

## 失败处理

- 下载失败：记录网络错误，尝试设置 `HF_ENDPOINT`，不要重装系统环境。
- 模型加载失败：记录 Transformers、torch、CUDA 版本，按 `ENV_SETUP.md` 回退版本。
- 格式检查失败：先修数据构造脚本或 schema alias，不要跳过 `check_format.py` 直接训练。
- 训练 OOM：先降 `MAX_LENGTH`、`BATCH_SIZE`，或增 `GRAD_ACCUM`，再考虑 DeepSpeed zero3。
- 输出格式差：提高格式规范样本比例，检查 assistant-only loss 是否生效。


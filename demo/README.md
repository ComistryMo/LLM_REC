# LLaMA-Factory 本地资源微调说明

> **demo**：仅含 10 条从原始 parquet 采样的未处理样本（`demo/baseline-data/baseline_data/sampled/sampled.parquet`），用于端到端跑通 pipeline。

## 1. 总览

| 项 | 值 |
|---|---|
| 基座模型 | `<YOUR_BASE_MODEL_PATH>` OneReason-0.8B-pretrain-competition |
| demo 数据 | `demo/baseline-data/baseline_data/sampled/sampled.parquet`|
| 训练框架 | LLaMA-Factory `0.9.6.dev0`|
| 微调方式 | 全量 SFT, bf16, packing+neat_packing, FlashAttention-2, Liger Kernel |
| 输出 | `demo/output/onereason_0.8b_sft/`|

## 2. 训练流程

```bash
bash demo/scripts/run_all.sh
```

`run_all.sh` 按顺序执行：

1. `00_install.sh` — 构建环境
2. `01_convert_data.sh` — `convertv2.py` 把 parquet → Alpaca JSONL
3. `02_register_dataset.py` —  注册数据
4. `03_train.sh` — `llamafactory-cli train demo/config/demo.yaml` 启动训练


万擎官网SFT数据集下载方式为：【模型服务】-【数据管理】-【数据集】-【下载比赛数据集】；
可解压到 demo/baseline-data/baseline_data/sampled下，对数据注册后完成训练

下载得到的是 jsonl 格式（`[{system, prompt, response}]`），需要用 `convert_jsonl.py` 转成 Alpaca jsonl：

```bash
# 把 sampled/ 下所有 *.jsonl 合并 → demo/data/data_final.jsonl
python demo/convert_jsonl.py \
  --input  demo/baseline-data/baseline_data/sampled \
  --output demo/data/data_final.jsonl \
  --shuffle --shuffle-seed 2026
```

## 3. 目录结构

```
demo/
├── README.md                       # 本文件
├── convertv2.py                    # parquet → Alpaca JSONL 转换器
├── convert_jsonl.py                # jsonl ({system,prompt,response}) → Alpaca JSONL 转换器
├── baseline-data/
│   └── baseline_data/
│       └── sampled/
│           └── sampled.parquet     # 10 条采样原始 parquet
├── config/
│   └── demo.yaml                   # 训练配置
├── data/
│   └── data_final.jsonl            # 10 条 alpaca 样本
├── scripts/
│   ├── 00_install.sh
│   ├── 01_convert_data.sh
│   ├── 02_register_dataset.py
│   ├── 03_train.sh
│   └── run_all.sh
└── output/                         # 训练产物 (运行后生成)
    └── onereason_0.8b_sft/
        ├── model.safetensors
        ├── trainer_state.json
        ├── training_loss.png
        └── runs/                   # tensorboard
```

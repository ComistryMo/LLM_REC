# Itemic 能力修复与全量转换实验（2026-07-02）

## 结论

当前最稳妥的候选模型是：

```text
/data/hz/models/OneReason-0.8B-base10-allfull90
```

它由官方基础模型与现有全参模型进行全参数线性插值得到：

```text
candidate = 0.10 * base + 0.90 * allfull
```

该候选不改变模型结构。相对现有 `allfull`，它改善 itemic、懂物料和懂世界代理指标，懂用户基本持平，懂推荐 token accuracy 下降 0.55 个百分点。提交前仍应以官方评测为准，不应把 teacher-forced 指标当成最终榜单分数。

不要继续使用 `material_2ep_allfull_gpu0/.../checkpoint-8000`。该分支已经把 domain 前缀学到 100%，但没有修复 a/b/c SID，继续训练更可能扩大能力偏移。

## 固定实验条件

- itemic 验证集：`ep1_valid.jsonl`，655 条，无训练样本重叠。
- 起点模型：`OneReason-0.8B-allfull-material2-userrec1-1k`。
- 小实验：全参数、BF16、300 optimizer steps、学习率 `3e-7`、batch 1、gradient accumulation 4。
- itemic 指标：分别统计 domain/a/b/c 的 teacher-forced loss、top-1、top-5，以及 a/b/c exact。
- 任务回归：固定哈希抽样，assistant-only teacher-forced token loss/accuracy。

## 能力退化定位

| 模型 | abc loss | abc top-1 | abc top-5 | domain top-1 |
| --- | ---: | ---: | ---: | ---: |
| 官方 base | 3.7888 | 0.2611 | 0.4875 | 0.1939 |
| allfull | 4.2143 | 0.2305 | 0.4300 | 0.7924 |
| material checkpoint-8000 | 4.2027 | 0.2377 | 未作为候选 | 1.0000 |

现有 SFT 明显增强了 domain/格式能力，但损伤了 a/b/c SID。仅恢复 `embed_tokens`、仅恢复 `lm_head`、同时恢复两者或 50% 混合对应 token 行均无效，说明问题不局限于 itemic 词向量或输出头，而是 transformer 内部表示发生了全局偏移。

## 数据重配小实验

从 20,000 个无泄漏物料样本构造三套等规模数据：

- `direct`：完整 domain+a+b+c。
- `sid_only`：给定 domain，只预测 a+b+c。
- `enhanced`：50% full、20% sid、a/b/c 分层任务各 10%。

| 配方 | abc loss | 相对 allfull | abc top-5 变化 |
| --- | ---: | ---: | ---: |
| direct 300 steps | 4.2173 | +0.0030 | +0.0031 |
| sid-only 300 steps | 4.2133 | -0.0010 | +0.0056 |
| enhanced 300 steps | 4.2142 | -0.0001 | +0.0015 |

低学习率短程训练下，直接重复完整 itemic 没有效果；SID-only 方向略正，但量级不足。下一轮若继续训练，应优先实现 assistant token 级加权 loss，提高 a/b/c 权重，并混入用户/推荐 replay，而不是继续堆完整 itemic 样本。

## 全参数插值

下表 `alpha` 表示 allfull 的参数占比：`base + alpha * (allfull - base)`。

| alpha | abc loss | abc top-1 | abc top-5 | 推荐 accuracy | 用户 accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 3.8196 | 0.2611 | 0.4921 | 0.5002 | 0.6020 |
| 0.50 | 3.9065 | 0.2580 | 0.4819 | 0.5849 | 0.6303 |
| 0.75 | 4.0474 | 0.2519 | 0.4601 | 0.6456 | 0.6446 |
| 0.90 | 4.1605 | 0.2341 | 0.4407 | 0.6747 | 0.6475 |
| 1.00, allfull | 4.2143 | 0.2305 | 0.4300 | 0.6802 | 0.6483 |

`alpha=0.90` 是保守 Pareto 点：

| 代理任务 | allfull loss / accuracy | alpha=0.90 loss / accuracy | 判断 |
| --- | --- | --- | --- |
| 懂物料 | 2.7704 / 0.4702 | 2.6991 / 0.4738 | 改善 |
| 懂用户 | 1.6422 / 0.6483 | 1.6285 / 0.6475 | 基本持平 |
| 懂推荐 | 1.2432 / 0.6802 | 1.2641 / 0.6747 | 小幅回退 |
| 懂世界 | 2.0547 / 0.5553 | 2.0096 / 0.5612 | 改善 |

`alpha=0.75` 可作为更激进的备选，但推荐 accuracy 下降 3.46 个百分点，未获得官方分数前不建议替换保守候选。

## 数据转换容量

原始五类 parquet 共 63,044,706 行、17.18 GB。只有 `OneReason_General` 带官方转换器需要的 `messages` 字段：

- 输入：158 个 parquet，152,005 行，1.98 GB。
- 输出：`official_stream_v1/competition_general.jsonl`。
- 结果：152,005/152,005 写入，4,051,723,288 bytes，无丢弃。
- 转换采用 batch streaming，不把完整数据集放入内存。
- 已注册为 LLaMA-Factory 数据集 `competition_general`。

其余四类不是可直接套用 `convertv2.py` 的对话数据：

| 数据 | 行数 | 处理要求 |
| --- | ---: | --- |
| Pid2Caption | 21,061,327 | 与 SID/物料元数据 join 后构造任务 |
| Pid2Sid | 35,914,095 | 作为 PID 到 itemic SID 映射表 |
| Pid2Tag | 5,417,279 | 与物料表 join 后构造标签理解任务 |
| UserProfile | 500,000 | 构造懂用户/懂推荐任务并做时间防泄漏 |

服务器当前仅剩约 120 GB。把四类数据机械地转成 JSONL 在物理上可能放得下，但没有训练语义，而且 join/多任务展开后可能超过安全余量。正确做法是按任务流式 join、边构造边写分片，不生成四份无用途的全量中间 JSONL。

## 复现命令

```bash
cd /data/hz/llmrec_competition
export PYTHONPATH=src
VENV=demo/LLaMA-Factory/.venv/bin/python

$VENV scripts/data/build_itemic_enhancement.py \
  --input /data/hz/onereason_competition/data/material_2ep/swift_messages_v2/ep2_train.jsonl \
  --validation /data/hz/onereason_competition/data/material_2ep/swift_messages_v2/ep1_valid.jsonl \
  --output-dir /data/hz/onereason_competition/data/itemic_pilot_v1

$VENV demo/convertv2_streaming.py \
  --input /data/hz/onereason_competition/hf_raw/data/OneReason_General \
  --output /data/hz/onereason_competition/data/official_stream_v1/competition_general.jsonl \
  --summary /data/hz/onereason_competition/data/official_stream_v1/competition_general_summary.json

CUDA_VISIBLE_DEVICES=3 $VENV scripts/eval/itemic_teacher_forced.py \
  --validation /data/hz/onereason_competition/data/material_2ep/swift_messages_v2/ep1_valid.jsonl \
  --model candidate=/data/hz/models/OneReason-0.8B-base10-allfull90 \
  --device cuda:0 --output logs/itemic_candidate.json
```

## 验收

- `pytest tests/`：13 passed。
- `pre_submit_check.py`：passed。
- GPU 模型加载：passed，`Qwen3ForCausalLM`。
- tokenizer：176,253 tokens，chat template 存在，四类 itemic probe 均为四个原子 token。
- 结构：hidden size、层数、attention heads、KV heads、vocab、tie embeddings 等关键字段与官方 base 一致。


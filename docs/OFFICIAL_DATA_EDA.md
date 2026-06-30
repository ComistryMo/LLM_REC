# Official Data EDA

分析日期：2026-07-01。原始归档 SHA-256：`13a7220febc949587ed15ade52562b0d9a04056351b645615ba749a3a360827b`。

## 数据概况

- 共 32,480 条 SFT 样本，来自 12 个 JSONL 文件；每一行是包含一个对象的 JSON 数组。
- 懂推荐 19,204 条，懂物料 10,384 条，懂用户 2,892 条；当前数据不含独立的懂世界样本。
- domain 分布：mixed 21,300，product 3,504，video 3,491，ad 3,385，live 800。
- 全部样本均包含 `<think>` 推理内容，原始 prompt 也均包含 `/think`；第一版 baseline 保持官方格式，不主动剥离。
- itemic token 合法率 100%，无空消息、非法 itemic 片段或疑似凭据。

## 长度与行为

- 输入 token P50/P90/P99：926 / 2,224 / 6,223.21。
- 输出 token P50/P90/P99：612 / 833 / 1,025。
- 1,893 条超过 4,096 token，40 条超过 8,192 token，最大 10,543 token；均来自懂用户。
- 懂用户历史长度 P50/P90/P99：192 / 293 / 365 个事件。
- 原子行为 Top 5：点击 208,502、长播 185,031、购买 83,672、深度转化 48,387、关注 47,993。

## 数据质量结论

- 存在 145 条精确重复，形成 113 个重复组，最大组大小为 4。训练/验证切分前应去重，避免验证泄漏。
- 高频 itemic token 集中明显，需按来源和 domain 分层验证，避免热门 token 让离线指标虚高。
- 8,192 token 是第一版 A100 baseline 的合理长度：仅过滤 40 条样本，同时保留大多数长用户历史。4,096 token 会截断 1,893 条，不建议作为无分析的默认值。
- 该数据覆盖懂物料、懂用户、懂推荐，但不直接补充懂世界。后续如加入外部通用数据，应单独控制比例并做遗忘评估。

## 已生成产物

- `data/processed/sft_unified.jsonl`：32,480 条统一 messages 数据。
- `data/processed/train_official_v1.jsonl`：31,648 条，去重且过滤超过 8,192 token 的样本。
- `data/processed/valid_official_v1.jsonl`：647 条，按 12 个来源分层抽取 2%。
- `data/processed/llamafactory_official_v1_*_sharegpt.jsonl`：LLaMA-Factory 副本。
- `data/eda/EDA_REPORT.md` 与 `data/eda/OFFICIAL_SPLIT_REPORT.json`：服务器上的完整统计。

切分使用固定 seed `2026`，训练集和验证集不存在精确内容重复。两份切分均通过 8,192 token 格式检查。

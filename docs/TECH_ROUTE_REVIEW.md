# 技术路线审阅

## 本地 docx 草稿总结

草稿的主线是：基于官方 SFT 数据和 50w 用户行为数据，构造 `D_material / D_user / D_rec / D_world` 四类训练集，通过数据配比和分阶段 SFT 提升 OneReason 在物料理解、用户需求理解、推荐能力和通用知识上的综合表现。

草稿强调：

- 数据组织优先于模型结构创新。
- 用户行为按时间构造成历史序列，避免未来信息泄漏。
- 推荐输出要严格控制 itemic token 或 JSON 格式。
- LoRA 时 attention/MLP 线性层为主要 target，最好关注 `embed_tokens` 和 `lm_head`。
- 本地验证集按四类能力拆分，不依赖隐藏评测反馈。

## 可信观点

- “提交 CKPT 而不是预测文件”意味着输出格式必须在训练阶段学会，不能依赖赛后后处理。
- 四类能力拆分与 OneReason 报告中的 R0/R1/R2/R3 推荐推理能力方向一致。
- 用户行为样本必须按时间切分；随机切分会泄漏未来行为。
- 先跑官方 SFT baseline，再加入 next-item 和用户需求演化样本，是风险最低的实验顺序。
- 最后一阶段混入通用知识样本，有助于降低推荐数据导致的通用能力遗忘。

## 需要验证或修正的观点

- 草稿中的 itemic token 示例使用 `<a_*>/<b_*>/<c_*>`，工程中统一规范为 `<s_a_*>/<s_b_*>/<s_c_*>`。
- 草稿使用 `<|live_begin|>`，工程中统一为 `<|living_begin|>`，并兼容清洗旧写法。
- 50w 行为数据字段名仍需等正式数据确认；当前通过 alias YAML 兼容。
- 外部数据虽可作为后续实验，但第一版 baseline 默认只使用官方数据，避免复现材料和规则风险。
- Transformers 版本存在文档冲突：FAQ 提到 v5.3.0，开发机指南示例为 4.56.2，模型卡最低要求为 4.51.0。工程脚本优先 5.3.0，失败后回退到可加载模型的稳定版本并记录。

## 修正后的技术路线

1. 原始模型 sanity：加载 tokenizer/model，检查 chat template 和 itemic token tokenize。
2. 官方 SFT 解析：统一为 messages JSONL，打 `task_type/output_type/domain` 标签。
3. 行为数据解析：按 `user_id,timestamp` 排序，保留 action、domain、query、itemic token。
4. 历史样本构造：先做 next-item、目标域推荐、用户需求摘要，再逐步扩展跨域和去噪任务。
5. 多源配比：默认 `material/user/rec/world = 25/25/30/20`，固定 seed，可复现拆分。
6. 训练顺序：官方 SFT baseline -> SFT + next-item -> SFT + 用户理解 -> 分阶段 SFT。
7. 本地验证：格式合法率、itemic 合法率、domain 命中、exact match/pass@k 和人工抽查。
8. 导出检查：模型可加载、配置完整、不含数据和敏感信息。

## 前期优先级

1. 跑通模型下载、加载、LoRA 训练、导出、提交前检查。
2. 等官方数据开放后第一时间做 EDA 和格式校验。
3. 建立四类 valid split，避免只盯推荐指标。
4. 优先提升 `D_rec` 样本质量，而不是过早堆复杂训练策略。
5. 分阶段训练作为后续冲榜策略，不阻塞第一版可提交模型。


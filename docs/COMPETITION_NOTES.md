# 比赛规则笔记

## 已确认信息

- 官方基础模型：`OpenOneRec/OneReason-0.8B-pretrain-competition`。
- 任务目标：基于官方模型微调并提交模型 CKPT 参与闭源评测。
- 数据方向：官方 SFT 数据与用户行为数据；行为数据用于构造用户理解和推荐样本。
- 四维评测：懂物料、懂用户、懂推荐、懂世界。
- 每日正式评测次数有限，当前记录为每日 3 次。
- 不允许修改模型结构；需要基于指定预训练 checkpoint 微调。
- LoRA 与全参微调均可作为训练路径；提交文件需满足平台要求。
- 外部数据可作为实验选项，但需要满足复现要求，提交时应提供数据来源、处理脚本和可复现说明。

## 待正式页面复核

- 官网 overview、dataset demo、metric 细节。
- 官方数据字段名、文件格式、样本规模和下载路径。
- 提交 CKPT 目录结构、压缩格式、大小限制和上传入口。
- 是否存在额外的外部数据白名单/黑名单。
- 是否对训练框架、Transformers 版本、推理 dtype 有硬性要求。

## 模型与 itemic token

工程内统一合法 itemic token：

```text
<|video_begin|><s_a_xxxx><s_b_xxxx><s_c_xxxx>
<|prod_begin|><s_a_xxxx><s_b_xxxx><s_c_xxxx>
<|ad_begin|><s_a_xxxx><s_b_xxxx><s_c_xxxx>
<|living_begin|><s_a_xxxx><s_b_xxxx><s_c_xxxx>
<|sid_begin|><s_a_xxxx><s_b_xxxx><s_c_xxxx>
```

兼容清洗：

- `<|live_begin|>` -> `<|living_begin|>`
- `<a_x>/<b_x>/<c_x>` -> `<s_a_x>/<s_b_x>/<s_c_x>`
- 删除 begin/end 成对示例中的 end token。

## Baseline 记录

- Baseline 1：只用官方 SFT 数据，目标是跑通完整训练与提交。
- Baseline 2：官方 SFT + 用户行为 next-item 样本，目标是提升懂推荐。
- Baseline 3：官方 SFT + 用户需求演化 + next-item，目标是提升懂用户和懂推荐。
- Baseline 4：分阶段训练，先物料/通用，再用户/推荐，最后四维均衡修正。

## 参考链接

- 比赛官网：https://ks-llmrec.streamlake.com/
- 万擎说明：https://www.streamlake.com/document/WANQING/mq57afym1d7p20atnau
- 万擎开发机指南：https://www.streamlake.com/document/WANQING/mh1g8b8aunh8esspfm
- 模型卡：https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition
- OneReason 技术报告：https://arxiv.org/abs/2606.06260


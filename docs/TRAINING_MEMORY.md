# Training Memory Estimate

实测日期：2026-07-01。硬件为 A100-SXM4-80GB，PyTorch 2.6.0+cu124，ms-swift 4.3.2，BF16、SDPA、gradient checkpointing、batch size 1。

## LoRA 实测

配置：rank 32、alpha 64、all-linear，实际可训练参数 20.1851M，占总参数 2.4567%。

| 实际 token 长度 | nvidia-smi 新增峰值 | Swift 自报峰值 | 单步时间 |
|---:|---:|---:|---:|
| 221 | 2,376 MiB | 2.18 GiB | 1.0 s |
| 3,325 | 4,378 MiB | 3.77 GiB | 1.2 s |
| 6,714 | 5,456 MiB | 4.82 GiB | 1.7 s |

8,192 token 预计新增约 6-7 GiB。考虑 CUDA context、评估、保存和样本差异，实际运行按 7-9 GiB 预算，并至少预留 12 GiB 空闲显存。

## 全参实测与估算

全参模型有 801.4336M 可训练参数。单卡 ZeRO-2 在仅 221 token 时已占用约 8.60 GiB，并在 optimizer step 额外申请 2.99 GiB，因此 10.9 GiB 空闲显存下 OOM。

- 1K 全参建议至少预留 14-16 GiB。
- 8K 全参估算约 15-20 GiB，稳妥按 24 GiB 以上空闲显存规划。
- 多卡 ZeRO-2 会分片梯度和优化器状态，但每张卡仍需容纳模型与激活；正式运行前应再做 8K、2-step smoke。

## 推荐方案

第一版使用单卡 LoRA 8K：`BATCH_SIZE=1`、`GRAD_ACCUM=16`。当前数据 P99 接近 8K，改成 4K 会截断约 1,893 条样本。

31,648 条训练数据在 batch size 1 下仍需要 31,648 个 micro-step；根据 smoke 速度，一轮 LoRA 约需 9-12 小时，GPU 被共享时会更慢。首次 Swift tokenization 受远端旧 CPU 影响明显，脚本默认 `DATASET_NUM_PROC=4`。

```bash
CUDA_VISIBLE_DEVICES=0 MAX_LENGTH=8192 BATCH_SIZE=1 GRAD_ACCUM=16 \
  bash scripts/train/train_swift_lora.sh
```

正式训练前可先跑 10-step：

```bash
CUDA_VISIBLE_DEVICES=0 MAX_LENGTH=8192 MAX_STEPS=10 REPORT_TO=none \
  bash scripts/train/train_swift_lora.sh
```

服务器当前 GPU 均被其他任务占用；完整训练应等待目标卡至少空出 12 GiB，最好独占一张卡。全参训练建议等待至少一张卡空出 24 GiB，或使用多卡 ZeRO-2。

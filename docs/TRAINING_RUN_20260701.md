# Multi-GPU staged training run (2026-07-01)

## Resource decision

- GPUs 2 and 3 were protected and never selected by any training command.
- Other users occupied about 73-74 GiB on each available A100 80GB, leaving only 6.8-8.2 GiB per card.
- A BF16 full-parameter smoke test was therefore unsafe. The production path used two-card LoRA/QLoRA with a process-scoped memory watchdog.
- The watchdog terminated only this repository's training process tree after two consecutive low-memory samples. No CUDA OOM occurred.

## Stage 1: material understanding

- Data: 10,176 train / 208 validation samples.
- Method: two-card BF16 LoRA, rank 32, all linear layers, max length 2,048.
- Result: completed 636/636 steps in 17m45s.
- Final train loss: 1.913.
- Validation loss: 1.711; validation token accuracy: 0.5697.
- Adapter: `outputs/swift_lora_stage1_material/v0-20260701-012142/checkpoint-636`.
- CPU-merged model: `/data/hz/models/OneReason-0.8B-stage1-material-merged`.

## Stage 2: user and recommendation

The rank-32 Stage 1 adapter was merged first. Stage 2 then used 4-bit QLoRA with rank 8 on `q_proj` and `v_proj`, reducing trainable parameters to 1.1469M (0.1969%).

1. A 640-token segment completed 150 optimization steps before the watchdog stopped a later memory spike. At step 150, validation loss was 1.792 and validation token accuracy was 0.5897.
2. Training continued from that adapter on the 512-token high-confidence subset. It completed another 125 optimization steps before the next protected stop. At step 125, validation loss was 1.670 and validation token accuracy was 0.6248.
3. The final Stage 2 adapter is `outputs/swift_qlora_stage2_user_rec_512_continued_safe/v0-20260701-031358/checkpoint-125`.

The 512-token continuation contains 6,399 train / 92 validation records. Longer samples remain untouched in the original processed datasets and should be revisited when exclusive GPU memory is available.

## Final artifact

- Merged model: `/data/hz/models/OneReason-0.8B-stage2-user-rec-final`.
- Size: about 1.6 GiB (`model.safetensors` about 1.5 GiB).
- `pre_submit_check.py`: passed.
- CPU `AutoTokenizer` and `AutoModelForCausalLM` load: passed.
- Loaded class: `Qwen3ForCausalLM`; vocabulary size: 176,253; chat template present.
- The final `config.json` is byte-identical to the official base config (SHA-256 `5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4`).

This artifact contains the completed material stage followed by 275 protected optimization steps on user/recommendation data. It is a safe baseline, not a claim that the full Stage 2 epoch completed.

## Full-parameter Stage 2 rerun

GPU 0 later became available with enough headroom, so Stage 2 was rerun as a complete single-GPU full-parameter SFT from the Stage 1 material-merged model.

- Data: 22,443 train / 416 validation records after framework tokenization and truncation checks.
- Method: 801.4336M trainable parameters (100%), BF16, DeepSpeed ZeRO-2, gradient checkpointing, max length 1,024, batch size 1, gradient accumulation 16.
- Device: GPU 0 only. GPUs 1-5 were not selected for this run.
- Result: completed 1,403/1,403 steps (one epoch) in 1h17m59s without OOM or watchdog intervention.
- Peak model memory reported by Swift: 21.43 GiB.
- Final train loss: 1.284.
- Final validation loss: 1.161; validation token accuracy: 0.6891.
- Training checkpoint: `outputs/swift_full_stage2_user_rec_1k_gpu0/v0-20260701-093202/checkpoint-1403`.
- Submission model: `/data/hz/models/OneReason-0.8B-stage2-user-rec-full-1k`.
- Submission model passed `pre_submit_check.py` and CPU loading; its `config.json` is byte-identical to the official base config.

This full-parameter model supersedes the earlier protected QLoRA baseline for evaluation.

## All-full rerun: two material epochs

A second clean lineage was trained without LoRA or QLoRA at any stage:

```text
official base -> material full SFT epoch 1 -> material full SFT epoch 2 -> user/rec full SFT epoch 1
```

- Device: GPU 0 only; BF16, DeepSpeed ZeRO-2, gradient checkpointing.
- Material epoch 1: batch 4, gradient accumulation 4, 636/636 steps, peak 21.61 GiB, train loss 1.901, validation loss 1.567, validation token accuracy 0.5721.
- Material epoch 2: batch 4, gradient accumulation 4, 636/636 steps, peak 21.61 GiB, train loss 1.586, validation loss 1.570, validation token accuracy 0.5732.
- User/rec epoch: batch 2, gradient accumulation 8, 1,403/1,403 steps, peak 22.36 GiB, train loss 1.286, validation loss 1.160, validation token accuracy 0.6891.
- No CUDA OOM occurred and the 8 GiB process watchdog did not intervene.
- Final checkpoint: `outputs/swift_allfull_v2_stage2_user_rec_gpu0/v0-20260701-122054/checkpoint-1403`.
- Submission model: `/data/hz/models/OneReason-0.8B-allfull-material2-userrec1-1k`.
- Submission checks and CPU loading passed; the final `config.json` is byte-identical to the official base config.

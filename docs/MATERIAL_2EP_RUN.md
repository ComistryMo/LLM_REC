# Material 2EP full-parameter run

## Data lineage

Source data follows `懂物料_2EP数据组织与训练交接文档.md`:

- EP1: `/data/hz/onereason_competition/data/material_2ep/ep1_think/懂物料.jsonl`
- EP2: `/data/hz/onereason_competition/data/material_2ep/ep2_no_think/懂物料.jsonl`
- Swift data: `/data/hz/onereason_competition/data/material_2ep/swift_messages_v2`

`prepare_material_2ep.py` groups by complete itemic token before splitting. This keeps both directions of one item, and the corresponding EP1/EP2 records, on the same side of the train/validation boundary.

Prepared counts:

| Stage | Train | Validation |
| --- | ---: | ---: |
| EP1 think/official-preserved | 678,978 | 1,309 |
| EP2 no-think direct | 339,821 | 655 |

EP1 contains 675,097 `/think` records and 5,190 old official `/no_think` records with an empty `<think>` block. They are retained without rewriting, as required by the handoff document's official-data preservation rule.

## Active configuration

- Physical device: GPU 0 only.
- Model: `/data/hz/models/OneReason-0.8B-pretrain-competition`.
- Method: full-parameter BF16 SFT, DeepSpeed ZeRO-2, gradient checkpointing.
- Max length: 1,024.
- Per-device batch size: 16; gradient accumulation: 1.
- EP1 learning rate: `1e-5`; EP2 learning rate: `5e-6`.
- One epoch per stage; EP2 starts from the final EP1 full checkpoint.
- Save and evaluate every 1,000 optimizer steps.
- `save_total_limit=2`.
- Output root: `outputs/material_2ep_allfull_gpu0`.
- Main log: `logs/train_material_2ep_allfull_gpu0.log`.
- Memory watchdog: GPU 0, 8,000 MiB minimum free memory.

The batch-16 smoke test reported 22.18 GiB peak model memory. The first production interval reported about 24.7 GiB and successfully wrote `checkpoint-1000` with model, Trainer, scheduler, RNG, and DeepSpeed optimizer state.

## Recovery

Resume EP1 from one of its retained checkpoints:

```bash
cd /data/hz/llmrec_competition
conda activate onereason-rec
CUDA_VISIBLE_DEVICES=0 RUN_STAGE=ep1 \
EP1_RESUME=/absolute/path/to/ep1_think/.../checkpoint-N \
bash scripts/train/train_material_2ep_full.sh
```

Start EP2 from the final EP1 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_STAGE=ep2 \
EP1_MODEL=/absolute/path/to/ep1_think/.../checkpoint-final \
bash scripts/train/train_material_2ep_full.sh
```

Resume EP2 while retaining the EP1 model lineage:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_STAGE=ep2 \
EP1_MODEL=/absolute/path/to/ep1_think/.../checkpoint-final \
EP2_RESUME=/absolute/path/to/ep2_no_think/.../checkpoint-N \
bash scripts/train/train_material_2ep_full.sh
```

# Official Demo Snapshot

- Source: <https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/tree/main/demo>
- Dataset revision at download: `8b88c769d74801df49d41e47cb8653b90cbe1015`
- Downloaded: `2026-07-02`
- Official framework: LLaMA-Factory `0.9.6.dev0`
- Official FlashAttention wheel SHA256: `22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113`

The files originally published under `demo/` are retained unchanged. Server-specific
configuration is additive:

- `config/server_a100.yaml`
- `config/server_smoke.yaml`
- `scripts/setup_server.sh`
- `scripts/prepare_server_data.sh`
- `scripts/register_server_dataset.py`
- `scripts/03_train_server.sh`

The previous ms-swift data engineering and training code is intentionally retained.
It is still required to reproduce existing checkpoints and provides EDA, validation,
evaluation, export, and recovery capabilities that are not included in the official demo.

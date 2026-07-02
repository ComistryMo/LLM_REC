#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

bash demo/scripts/00_install.sh
demo/LLaMA-Factory/.venv/bin/python demo/scripts/02_register_dataset.py

{
  echo "installed_at=$(date --iso-8601=seconds)"
  echo "llamafactory_commit=$(git -C demo/LLaMA-Factory rev-parse HEAD)"
  demo/LLaMA-Factory/.venv/bin/python - <<'PY'
from importlib.metadata import version
import flash_attn
import torch
import transformers

print(f"python_environment=demo/LLaMA-Factory/.venv")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"llamafactory={version('llamafactory')}")
print(f"flash_attn={flash_attn.__version__}")
print(f"liger_kernel={version('liger-kernel')}")
PY
} > demo/server_environment.txt

echo "[ok] server environment recorded in demo/server_environment.txt"

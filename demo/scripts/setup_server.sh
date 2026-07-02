#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

exec > >(tee -a demo/setup_server.log) 2>&1

BOOTSTRAP=demo/.bootstrap
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-/data/conda/envs/onereason-rec/bin/python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$PWD/demo/.uv-cache"}
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
export UV_INDEX_URL=${UV_INDEX_URL:-$UV_DEFAULT_INDEX}

if ! command -v uv >/dev/null 2>&1; then
  if [[ ! -x "$BOOTSTRAP/bin/uv" ]]; then
    echo "[run] bootstrap project-local uv"
    "$BOOTSTRAP_PYTHON" -m venv "$BOOTSTRAP"
    "$BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check "uv==0.8.22"
  fi
  export PATH="$PWD/$BOOTSTRAP/bin:$PATH"
fi

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

uv cache clean >/dev/null
echo "[ok] server environment recorded in demo/server_environment.txt"

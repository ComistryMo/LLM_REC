#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

exec > >(tee -a demo/setup_server.log) 2>&1

BOOTSTRAP=demo/.bootstrap
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-/data/conda/envs/onereason-rec/bin/python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$PWD/demo/.uv-cache"}
export UV_INDEX_URL=${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

if ! command -v uv >/dev/null 2>&1; then
  if [[ ! -x "$BOOTSTRAP/bin/uv" ]]; then
    echo "[run] bootstrap project-local uv"
    "$BOOTSTRAP_PYTHON" -m venv "$BOOTSTRAP"
    "$BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check "uv==0.8.22"
  fi
  export PATH="$PWD/$BOOTSTRAP/bin:$PATH"
fi

VENV=demo/LLaMA-Factory/.venv
if ! "$VENV/bin/python" -c "from importlib.metadata import version; assert version('llamafactory') == '0.9.6.dev0'; assert version('liger-kernel') == '0.8.0'" 2>/dev/null; then
  bash demo/scripts/00_install.sh
else
  echo "[skip] LLaMA-Factory and Liger core packages already installed"
fi

echo "[run] ensure PyTorch 2.7.1 cu126 runtime dependencies"
uv pip install --python "$VENV/bin/python" --no-deps \
  --default-index https://download.pytorch.org/whl/cu126 \
  "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" "torchaudio==2.7.1+cu126"

uv pip install --python "$VENV/bin/python" --no-deps \
  "sympy==1.13.3" \
  "nvidia-cuda-nvrtc-cu12==12.6.77" \
  "nvidia-cuda-runtime-cu12==12.6.77" \
  "nvidia-cuda-cupti-cu12==12.6.80" \
  "nvidia-cudnn-cu12==9.5.1.17" \
  "nvidia-cublas-cu12==12.6.4.1" \
  "nvidia-cufft-cu12==11.3.0.4" \
  "nvidia-curand-cu12==10.3.7.77" \
  "nvidia-cusolver-cu12==11.7.1.2" \
  "nvidia-cusparse-cu12==12.5.4.2" \
  "nvidia-cusparselt-cu12==0.6.3" \
  "nvidia-nccl-cu12==2.26.2" \
  "nvidia-nvtx-cu12==12.6.77" \
  "nvidia-nvjitlink-cu12==12.6.85" \
  "nvidia-cufile-cu12==1.11.1.6" \
  "triton==3.3.1"

uv pip install --python "$VENV/bin/python" "numpy==1.26.4" "tensorboard"

FLASH_WHEEL=$(find demo/wheels -maxdepth 1 -type f -name 'flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl' -print -quit 2>/dev/null || true)
if [[ -z "$FLASH_WHEEL" ]]; then
  echo "[error] official FlashAttention wheel is missing under demo/wheels" >&2
  exit 1
fi
EXPECTED_FLASH_SHA256=22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113
echo "$EXPECTED_FLASH_SHA256  $FLASH_WHEEL" | sha256sum --check
uv pip install --python "$VENV/bin/python" --no-deps "$FLASH_WHEEL"
uv pip check --python "$VENV/bin/python"

FA_PY="$VENV/lib/python3.11/site-packages/transformers/integrations/flash_attention.py"
if ! grep -q "s_aux=s_aux.to(query.dtype) if s_aux is not None else None" "$FA_PY"; then
  sed -i 's|s_aux=s_aux.to(query.dtype),|s_aux=s_aux.to(query.dtype) if s_aux is not None else None,|' "$FA_PY"
fi

demo/LLaMA-Factory/.venv/bin/python demo/scripts/02_register_dataset.py

{
  echo "installed_at=$(date --iso-8601=seconds)"
  echo "llamafactory_commit=$(git -C demo/LLaMA-Factory rev-parse HEAD)"
  demo/LLaMA-Factory/.venv/bin/python - <<'PY'
from importlib.metadata import version
import flash_attn
import numpy
import torch
import transformers

print(f"python_environment=demo/LLaMA-Factory/.venv")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"numpy={numpy.__version__}")
print(f"transformers={transformers.__version__}")
print(f"llamafactory={version('llamafactory')}")
print(f"flash_attn={flash_attn.__version__}")
print(f"liger_kernel={version('liger-kernel')}")
PY
} > demo/server_environment.txt

uv cache clean >/dev/null
echo "[ok] server environment recorded in demo/server_environment.txt"

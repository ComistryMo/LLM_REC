#!/usr/bin/env bash
# 00_install.sh - 安装 LLaMA-Factory + 加速包(liger-kernel + flash-attn)
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT=demo
LF_DIR=$ROOT/LLaMA-Factory
VENV=$LF_DIR/.venv

# 1.1 拉取 LLaMA-Factory
if [ -d "$LF_DIR/.git" ]; then
  echo "[skip] $LF_DIR already cloned"
else
  echo "[run] git clone LLaMA-Factory"
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
fi

# 1.2 创建 Python 3.11 venv (flash-attn wheel 是 cp311)
if [ -x "$VENV/bin/python" ]; then
  echo "[skip] venv already exists at $VENV"
else
  echo "[run] uv venv --python 3.11"
  uv venv --python 3.11 "$VENV"
fi
source "$VENV/bin/activate"

# 1.3 LLaMA-Factory + metrics
echo "[run] install LLaMA-Factory (editable) + metrics deps"
uv pip install -e "$LF_DIR"
uv pip install -r "$LF_DIR/requirements/metrics.txt"

# 1.4 配置环境（请自行配置）
echo "[run] pin torch 2.7.1+cu126"
SP="$VENV/lib/python3.11/site-packages"
uv pip uninstall torch torchvision torchaudio sympy 2>/dev/null || true
rm -rf "$SP/torch" "$SP/sympy" "$SP/functorch"
uv pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126
uv pip install --force-reinstall --no-deps "sympy==1.13.3"

echo "[run] install liger-kernel 0.8.0 + flash-attn 2.7.4.post1"
uv pip install --no-deps "liger-kernel==0.8.0"
uv pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" \
  > "$ROOT/flash_install.log" 2>&1 &
FA_PID=$!
echo "[run] flash-attn install backgrounded, pid=$FA_PID, log=$ROOT/flash_install.log"
wait $FA_PID
uv pip install tensorboard

echo "[run] patch transformers/integrations/flash_attention.py (s_aux None guard)"
FA_PY="$SP/transformers/integrations/flash_attention.py"
if grep -q "s_aux=s_aux.to(query.dtype) if s_aux is not None else None" "$FA_PY"; then
  echo "[skip] patch already applied"
else
  sed -i 's|s_aux=s_aux.to(query.dtype),|s_aux=s_aux.to(query.dtype) if s_aux is not None else None,|' "$FA_PY"
  echo "[ok] patched"
fi

# 检查环境
echo "[verify] versions"
python - <<'PY'
import torch, flash_attn, transformers, sympy
from importlib.metadata import version
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("flash_attn:", flash_attn.__version__)
print("transformers:", transformers.__version__)
print("liger-kernel:", version("liger-kernel"))
print("sympy:", sympy.__version__)
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
print("liger qwen3 hook: OK")
PY
echo "[ok] install finished"

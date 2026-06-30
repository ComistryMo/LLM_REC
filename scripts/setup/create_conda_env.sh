#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-onereason-rec}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1+cu121}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1+cu121}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1+cu121}"
WANDB_MODE="${WANDB_MODE:-disabled}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please run this on the Wanqing/dev machine with conda available." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi
conda activate "${ENV_NAME}"
python -m pip install -i "${PYPI_INDEX}" --upgrade pip setuptools wheel

conda install -y -n "${ENV_NAME}" filelock typing_extensions networkx jinja2 fsspec sympy pillow

python -m pip install --index-url "${CUDA_INDEX}" --no-deps \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

set +e
python -m pip install -i "${PYPI_INDEX}" "transformers==5.3.0"
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
PY
TRANSFORMERS_OK=$?
set -e

if [[ "${TRANSFORMERS_OK}" -ne 0 ]]; then
  echo "transformers==5.3.0 failed to import; falling back to 4.56.2 for model-card/dev-guide compatibility."
  python -m pip install -i "${PYPI_INDEX}" --force-reinstall "transformers==4.56.2"
fi

python -m pip install -i "${PYPI_INDEX}" \
  datasets accelerate peft trl \
  "ms-swift==4.3.2" "llamafactory==0.9.5" \
  modelscope huggingface_hub sentencepiece protobuf safetensors \
  jsonlines pandas "numpy==1.26.4" "polars-lts-cpu==1.33.1" pyarrow scikit-learn tensorboard pyyaml tqdm pytest

export WANDB_MODE="${WANDB_MODE}"
python - <<'PY'
import importlib
mods = ["torch", "transformers", "datasets", "accelerate", "peft", "swift", "llamafactory", "pandas", "polars"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{name}: FAILED {exc}")
        raise
PY

echo "Conda env ${ENV_NAME} is ready."

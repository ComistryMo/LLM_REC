#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

bash demo/scripts/00_install.sh
bash demo/scripts/01_convert_data.sh
demo/LLaMA-Factory/.venv/bin/python demo/scripts/02_register_dataset.py
bash demo/scripts/03_train.sh

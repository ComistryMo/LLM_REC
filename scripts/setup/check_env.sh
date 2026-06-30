#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/hz/llmrec_competition}"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/env_check_$(date +%Y%m%d_%H%M%S).log"

{
  echo "## nvidia-smi"
  nvidia-smi || true
  echo
  echo "## df -h"
  df -h
  echo
  echo "## free -h"
  free -h
  echo
  echo "## conda --version"
  conda --version || true
  echo
  echo "## python --version"
  python --version || true
  echo
  echo "## git --version"
  git --version || true
} | tee "${LOG_FILE}"

echo "Environment check written to ${LOG_FILE}"


#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmrec.data_schema import SECRET_RE


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-submit model directory checks.")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Missing model dir: {model_dir}")
    names = {p.name for p in model_dir.iterdir()}
    required_any = [{"config.json"}, {"adapter_config.json"}]
    if not any(group & names for group in required_any):
        raise SystemExit("No config.json or adapter_config.json found.")
    if not any(p.suffix in {".safetensors", ".bin"} for p in model_dir.rglob("*")):
        raise SystemExit("No model or adapter weight file found.")
    forbidden_dirs = {"data", "raw", "processed", "interim", "wandb"}
    bad_dirs = [str(p) for p in model_dir.rglob("*") if p.is_dir() and p.name in forbidden_dirs]
    if bad_dirs:
        raise SystemExit(f"Forbidden data-like directories inside model export: {bad_dirs[:5]}")
    text_files = [p for p in model_dir.rglob("*") if p.suffix.lower() in {".json", ".md", ".txt", ".yaml", ".yml"}]
    for path in text_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if SECRET_RE.search(text):
            raise SystemExit(f"Secret-like text found in {path}")
        if path.suffix == ".json":
            json.loads(text)
    print(f"Pre-submit check passed: {model_dir}")


if __name__ == "__main__":
    main()


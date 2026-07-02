#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="server_data_final")
    parser.add_argument("--data", default="demo/data/server_data_final.jsonl")
    parser.add_argument("--dataset-info", default="demo/LLaMA-Factory/data/dataset_info.json")
    args = parser.parse_args()

    info_path = Path(args.dataset_info).resolve()
    data_path = Path(args.data).resolve()
    if not data_path.is_file() or data_path.stat().st_size == 0:
        raise SystemExit(f"dataset is missing or empty: {data_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[args.name] = {
        "file_name": str(data_path),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
        },
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] registered {args.name}: {data_path}")


if __name__ == "__main__":
    main()

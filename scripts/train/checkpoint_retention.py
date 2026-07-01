#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import time

import psutil


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def complete_checkpoint(path: Path) -> bool:
    if not (path / "model.safetensors").is_file() or not (path / "trainer_state.json").is_file():
        return False
    step = path.name.removeprefix("checkpoint-")
    state_dir = path / f"global_step{step}"
    return state_dir.is_dir() and any(state_dir.glob("*optim_states.pt"))


def prune_stage(stage_root: Path, limit: int) -> None:
    resolved_root = stage_root.resolve()
    checkpoints = [
        path
        for path in stage_root.glob("v*/checkpoint-*")
        if path.is_dir() and complete_checkpoint(path)
    ]
    checkpoints.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in checkpoints[limit:]:
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise RuntimeError(f"Refusing to remove path outside stage root: {resolved}")
        log(f"removing old complete checkpoint: {resolved}")
        shutil.rmtree(resolved)


def prune(output_root: Path, limit: int) -> None:
    for stage in ("ep1_think", "ep2_no_think"):
        stage_root = output_root / stage
        if stage_root.is_dir():
            prune_stage(stage_root, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep the newest complete checkpoints across Swift version dirs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    pid = int(args.pid_file.read_text().strip()) if args.pid_file else None
    log(f"watching output_root={output_root} limit={args.limit} pid={pid}")
    while pid is None or psutil.pid_exists(pid):
        prune(output_root, args.limit)
        if pid is None:
            return
        time.sleep(args.interval)
    prune(output_root, args.limit)
    log("training process exited; final pruning complete")


if __name__ == "__main__":
    main()

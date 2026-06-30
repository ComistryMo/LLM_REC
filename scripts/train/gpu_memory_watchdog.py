#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time

import psutil


def log(path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def free_memory_mib(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def terminate_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    processes = root.children(recursive=True) + [root]
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=15)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop one training tree before GPU memory is exhausted.")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, default=1800)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--consecutive", type=int, default=3)
    args = parser.parse_args()

    pid = int(args.pid_file.read_text(encoding="utf-8").strip())
    low_count = 0
    log(args.log, f"watching pid={pid} gpus={args.gpus} min_free_mib={args.min_free_mib}")
    while psutil.pid_exists(pid):
        free_by_gpu = {gpu: free_memory_mib(gpu) for gpu in args.gpus}
        minimum = min(free_by_gpu.values())
        low_count = low_count + 1 if minimum < args.min_free_mib else 0
        if low_count >= args.consecutive:
            log(args.log, f"threshold breached: free_mib={free_by_gpu}; terminating pid={pid}")
            terminate_tree(pid)
            return
        time.sleep(args.interval)
    log(args.log, f"pid={pid} exited; watchdog stopping")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely extract the official dataset archive.")
    parser.add_argument("archive", help="Path to dataset.tar.gz")
    parser.add_argument("--output", default="data/raw/official")
    parser.add_argument("--name-encoding", default="gbk")
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive, "r:gz", encoding=args.name_encoding) as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if member.issym() or member.islnk() or member.name.startswith("/") or ".." in parts:
                raise ValueError(f"Unsafe archive member: {member.name!r}")
            if not member.isfile():
                continue
            destination = (output / member.name).resolve()
            if output not in destination.parents:
                raise ValueError(f"Archive member escapes output directory: {member.name!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {member.name!r}")
            with destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
            print(f"extracted {member.name} ({member.size} bytes)")
    print(f"Extracted {extracted} files to {output}")


if __name__ == "__main__":
    main()

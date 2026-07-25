#!/usr/bin/env python3
"""One-shot official Pixal3D inference; shares the TRELLIS.2 worker contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--decimation-target", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inference = Path(args.repo) / "inference.py"
    if not inference.is_file():
        raise SystemExit(f"missing official Pixal3D inference entry point: {inference}")
    environment = os.environ.copy()
    environment.setdefault("ATTN_BACKEND", "sdpa")
    subprocess.run(
        [
            os.environ.get("OMNISERVE_3D_PIXAL_PYTHON", os.sys.executable),
            str(inference),
            "--image",
            args.image,
            "--output",
            args.output,
            "--low_vram",
            "--resolution",
            str(max(1024, args.resolution)),
        ],
        cwd=args.repo,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()

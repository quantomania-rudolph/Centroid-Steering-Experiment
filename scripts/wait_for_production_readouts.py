#!/usr/bin/env python
"""Wait for production lens, then generate per-layer readouts (8-26)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt",
    )
    parser.add_argument("--layers", default="8-26")
    parser.add_argument("--poll-sec", type=int, default=120)
    args = parser.parse_args()

    print(f"Waiting for production lens: {args.lens}")
    while not args.lens.exists():
        time.sleep(args.poll_sec)
        print(f"  still waiting ({args.lens.name})")

    print("Production lens found. Generating per-layer readouts...")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_all_layer_readouts.py"),
            "--lens",
            str(args.lens),
            "--layers",
            args.layers,
        ],
        cwd=str(ROOT),
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Download Qwen2.5-Coder-7B-Instruct weights to local cache / artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "artifacts" / "models",
    )
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        args.model_id,
        cache_dir=str(args.cache_dir),
        local_dir=str(args.cache_dir / "Qwen2.5-Coder-7B-Instruct"),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

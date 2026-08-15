#!/usr/bin/env python3
"""Download lens + metadata from GPU instance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = RENTAL_ROOT / "output"
sys.path.insert(0, str(RENTAL_ROOT / "scripts"))

from load_secrets import load_secrets, require  # noqa: E402


def scp_base(secrets: dict[str, str]) -> list[str]:
    cmd = [
        "scp",
        "-P",
        str(secrets["RENTAL_SSH_PORT"]),
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    key = secrets.get("RENTAL_SSH_KEY_PATH", "")
    if key:
        cmd.extend(["-i", key])
    return cmd


def main() -> int:
    secrets = require(
        ["RENTAL_SSH_HOST", "RENTAL_SSH_PORT", "RENTAL_SSH_USER"],
        load_secrets(),
    )
    remote_dir = secrets.get("RENTAL_REMOTE_DIR", "/workspace/jlens-rental")
    remote = f"{secrets['RENTAL_SSH_USER']}@{secrets['RENTAL_SSH_HOST']}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = ["qwen_jacobian_lens.pt", "fit_meta.json", "ckpt.pt", "fit_progress.json"]

    for name in files:
        remote_path = f"{remote}:{remote_dir}/{name}"
        local_path = OUTPUT_DIR / name
        cmd = scp_base(secrets) + [remote_path, str(local_path)]
        print(f"Downloading {name}...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            if name == "ckpt.pt":
                print(f"  (optional {name} not found — OK if fit finished)", flush=True)
                continue
            print(proc.stderr, flush=True)
            return proc.returncode
        print(f"  -> {local_path}", flush=True)

    lens = OUTPUT_DIR / "qwen_jacobian_lens.pt"
    if lens.is_file():
        print(f"Lens ready: {lens} ({lens.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

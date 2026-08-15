#!/usr/bin/env python3
"""SSH into GPU instance and run fit (streams output)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RENTAL_ROOT / "scripts"))

from load_secrets import load_secrets, require  # noqa: E402


def ssh_base(secrets: dict[str, str]) -> list[str]:
    cmd = [
        "ssh",
        "-p",
        str(secrets["RENTAL_SSH_PORT"]),
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    key = secrets.get("RENTAL_SSH_KEY_PATH", "")
    if key:
        cmd.extend(["-o", "IdentitiesOnly=yes", "-i", key])
    cmd.append(f"{secrets['RENTAL_SSH_USER']}@{secrets['RENTAL_SSH_HOST']}")
    return cmd


def main() -> int:
    secrets = require(
        ["RENTAL_SSH_HOST", "RENTAL_SSH_PORT", "RENTAL_SSH_USER"],
        load_secrets(),
    )
    remote_dir = secrets.get("RENTAL_REMOTE_DIR", "/workspace/jlens-rental")

    # Strip Windows CRLF if present (causes `set: pipefail\\r` on Linux bash).
    prep = (
        f"cd {remote_dir} && "
        "sed -i 's/\\r$//' run_fit.sh fit_on_gpu.py 2>/dev/null; "
        "chmod +x run_fit.sh && bash run_fit.sh"
    )
    fit_cmd = ssh_base(secrets) + [prep]
    print("Starting remote fit (this takes ~30-60 min on H100)...", flush=True)
    return subprocess.call(fit_cmd)


if __name__ == "__main__":
    raise SystemExit(main())

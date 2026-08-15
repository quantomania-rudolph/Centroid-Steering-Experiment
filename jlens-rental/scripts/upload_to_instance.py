#!/usr/bin/env python3
"""SCP rental bundle to GPU instance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RENTAL_ROOT / "scripts"))

from load_secrets import load_secrets, require  # noqa: E402

BUNDLE_ZIP = RENTAL_ROOT / "bundle" / "rental_bundle.zip"


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
        cmd.extend(["-o", "IdentitiesOnly=yes", "-i", key])
    return cmd


def main() -> int:
    if not BUNDLE_ZIP.is_file():
        print("Run: python scripts/prepare_bundle.py first", flush=True)
        return 1

    secrets = require(
        ["RENTAL_SSH_HOST", "RENTAL_SSH_PORT", "RENTAL_SSH_USER"],
        load_secrets(),
    )
    remote_dir = secrets.get("RENTAL_REMOTE_DIR", "/workspace/jlens-rental")

    remote = f"{secrets['RENTAL_SSH_USER']}@{secrets['RENTAL_SSH_HOST']}"
    mkdir_cmd = ssh_base(secrets) + [f"mkdir -p {remote_dir}"]
    print("Creating remote dir...", flush=True)
    subprocess.run(mkdir_cmd, check=True)

    scp_cmd = scp_base(secrets) + [str(BUNDLE_ZIP), f"{remote}:{remote_dir}/"]
    print("Uploading bundle...", flush=True)
    subprocess.run(scp_cmd, check=True)

    unzip_cmd = ssh_base(secrets) + [
        f"cd {remote_dir} && unzip -o rental_bundle.zip && "
        "sed -i 's/\\r$//' run_fit.sh fit_on_gpu.py 2>/dev/null; "
        "chmod +x run_fit.sh"
    ]
    print("Unpacking on instance...", flush=True)
    subprocess.run(unzip_cmd, check=True)
    print("Upload complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

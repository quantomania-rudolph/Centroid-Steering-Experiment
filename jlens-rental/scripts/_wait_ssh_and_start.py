#!/usr/bin/env python3
"""Wait for Vast instance SSH, then start full pipeline (no auto-destroy)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RENTAL_ROOT / "scripts"))
from load_secrets import load_secrets  # noqa: E402


def main() -> int:
    secrets = load_secrets()
    key = secrets["VAST_API_KEY"].strip()
    iid = secrets["VAST_INSTANCE_ID"].strip() or "46772095"
    pub = Path.home().joinpath(".ssh", "id_ed25519.pub").read_text(encoding="utf-8").strip()
    keypath = secrets.get("RENTAL_SSH_KEY_PATH") or str(Path.home() / ".ssh" / "id_ed25519")
    envp = RENTAL_ROOT / "secrets" / "secrets.env"

    def attach() -> None:
        body = json.dumps({"ssh_key": pub}).encode()
        req = urllib.request.Request(
            f"https://console.vast.ai/api/v0/instances/{iid}/ssh",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                print("attach", resp.read().decode()[:200], flush=True)
        except Exception as exc:
            print("attach", exc, flush=True)

    attach()
    for i in range(40):
        req = urllib.request.Request(
            f"https://console.vast.ai/api/v0/instances/{iid}/",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        inst = data.get("instances") or data
        if isinstance(inst, list):
            inst = inst[0]
        status = inst.get("actual_status")
        ip = inst.get("public_ipaddr") or "209.20.157.57"
        ports = inst.get("ports") or {}
        ssh = ((ports.get("22/tcp") or [{}])[0] or {}).get("HostPort")
        gpu = inst.get("gpu_name")
        print(f"t={i * 5}s status={status} ssh={ssh} gpu={gpu}", flush=True)
        if status == "running" and ssh:
            attach()
            lines = []
            for line in envp.read_text(encoding="utf-8").splitlines():
                if line.startswith("RENTAL_SSH_HOST="):
                    lines.append(f"RENTAL_SSH_HOST={ip}")
                elif line.startswith("RENTAL_SSH_PORT="):
                    lines.append(f"RENTAL_SSH_PORT={ssh}")
                elif line.startswith("VAST_INSTANCE_ID="):
                    lines.append(f"VAST_INSTANCE_ID={iid}")
                elif line.startswith("AUTO_DESTROY="):
                    lines.append("AUTO_DESTROY=0")
                else:
                    lines.append(line)
            envp.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
            proc = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=12",
                    "-o",
                    "IdentitiesOnly=yes",
                    "-i",
                    keypath,
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-p",
                    str(ssh),
                    f"root@{ip}",
                    "echo SSH_OK; nvidia-smi -L",
                ],
                capture_output=True,
                text=True,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            print(out[-1000:], flush=True)
            if "SSH_OK" not in out:
                time.sleep(5)
                continue
            print("SSH_READY — starting pipeline", flush=True)
            log = RENTAL_ROOT / "artifacts_pipeline.log"
            with log.open("w", encoding="utf-8") as fh:
                return subprocess.call(
                    [sys.executable, str(RENTAL_ROOT / "scripts" / "run_full_pipeline.py")],
                    cwd=str(RENTAL_ROOT),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                )
        time.sleep(5)
    print("TIMEOUT waiting for SSH", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

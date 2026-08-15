#!/usr/bin/env python3
"""Attach SSH key via Vast API, prove login, start full pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RENTAL_ROOT / "scripts"))
from load_secrets import load_secrets  # noqa: E402


def api(method: str, url: str, key: str, data: dict | None = None) -> dict | None:
    body = None if data is None else json.dumps(data).encode()
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode()
            print(f"{method} {url} -> {resp.status} {raw[:500]}", flush=True)
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"{method} {url} -> HTTP {exc.code} {detail[:500]}", flush=True)
        return None
    except Exception as exc:
        print(f"{method} {url} -> {exc}", flush=True)
        return None


def get_instance(key: str, iid: str) -> dict:
    data = api("GET", f"https://console.vast.ai/api/v0/instances/{iid}/", key) or {}
    inst = data.get("instances") or data
    if isinstance(inst, list):
        inst = inst[0]
    return inst


def try_ssh(ip: str, port: str, keypath: str) -> bool:
    cmd = [
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
        str(port),
        f"root@{ip}",
        "echo SSH_OK; nvidia-smi -L; ls /workspace | head",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-1200:], flush=True)
    return "SSH_OK" in out


def main() -> int:
    secrets = load_secrets()
    key = secrets["VAST_API_KEY"].strip()
    iid = secrets["VAST_INSTANCE_ID"].strip()
    pub = Path.home().joinpath(".ssh", "id_ed25519.pub").read_text(encoding="utf-8").strip()
    keypath = secrets.get("RENTAL_SSH_KEY_PATH") or str(Path.home() / ".ssh" / "id_ed25519")

    print("=== attach SSH key to instance ===", flush=True)
    api("POST", f"https://console.vast.ai/api/v0/instances/{iid}/ssh", key, {"ssh_key": pub})

    print("=== execute authorized_keys inject ===", flush=True)
    cmd = (
        "mkdir -p /root/.ssh; chmod 700 /root/.ssh; "
        f"echo '{pub}' >> /root/.ssh/authorized_keys; "
        "chmod 600 /root/.ssh/authorized_keys; "
        "sort -u /root/.ssh/authorized_keys -o /root/.ssh/authorized_keys; "
        "echo KEY_INSTALLED; wc -l /root/.ssh/authorized_keys"
    )
    res = api(
        "PUT",
        f"https://console.vast.ai/api/v0/instances/command/{iid}/",
        key,
        {"command": cmd},
    )
    if res and res.get("result_url"):
        time.sleep(5)
        try:
            with urllib.request.urlopen(res["result_url"], timeout=30) as resp:
                print("EXEC_RESULT:", resp.read().decode()[:800], flush=True)
        except Exception as exc:
            print("result_url fetch:", exc, flush=True)

    for attempt in range(12):
        inst = get_instance(key, iid)
        status = inst.get("actual_status")
        ip = inst.get("public_ipaddr") or secrets.get("RENTAL_SSH_HOST")
        ports = inst.get("ports") or {}
        ssh_port = ((ports.get("22/tcp") or [{}])[0] or {}).get("HostPort") or secrets.get(
            "RENTAL_SSH_PORT", "19717"
        )
        print(f"attempt {attempt}: status={status} ip={ip} port={ssh_port}", flush=True)
        if status == "running" and ip:
            # update secrets host/port
            env_path = RENTAL_ROOT / "secrets" / "secrets.env"
            text = env_path.read_text(encoding="utf-8")
            lines = []
            for line in text.splitlines():
                if line.startswith("RENTAL_SSH_HOST="):
                    lines.append(f"RENTAL_SSH_HOST={ip}")
                elif line.startswith("RENTAL_SSH_PORT="):
                    lines.append(f"RENTAL_SSH_PORT={ssh_port}")
                elif line.startswith("RENTAL_SSH_KEY_PATH="):
                    lines.append(f"RENTAL_SSH_KEY_PATH={keypath}")
                else:
                    lines.append(line)
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if try_ssh(ip, str(ssh_port), keypath):
                print("SSH READY — starting full pipeline", flush=True)
                return subprocess.call(
                    [sys.executable, str(RENTAL_ROOT / "scripts" / "run_full_pipeline.py")],
                    cwd=str(RENTAL_ROOT),
                )
        time.sleep(5)

    print("FAILED: could not SSH after attach/execute", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

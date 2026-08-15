#!/usr/bin/env python
"""One-line cleanup-run proof (token-minimal)."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "experiments" / "unicaa01_corpus" / "cleanup_retry.log"
PORT = 8767


def _server() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as resp:
            h = json.loads(resp.read().decode("utf-8"))
            return bool(h.get("ok")), str(h.get("device") or "?")
    except Exception as exc:
        return False, type(exc).__name__


def _cleanup_pids() -> list[int]:
    if sys.platform != "win32":
        return []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'run_unicaa01_corpus|run_corpus_cleanup' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def _log_tail() -> str:
    if not LOG.is_file():
        return "no_log"
    raw = LOG.read_bytes()
    text = raw.decode("utf-16", errors="replace") if raw.startswith(b"\xff\xfe") else raw.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("=== ") and line.endswith(" ==="):
            return line[4:-4]
        if line.startswith("=== RUN "):
            return line.replace("=== RUN ", "run:")
    return "?"


def main() -> int:
    srv_ok, srv_dev = _server()
    pids = _cleanup_pids()
    n_files = len(list((ROOT / "experiments/unicaa01_corpus/out").glob("*_latest.json")))
    working = bool(pids) and srv_ok
    proof = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "CLEANUP_WORKING": working,
        "cleanup_pids": pids,
        "srv": ("up:" if srv_ok else "down:") + srv_dev,
        "files": n_files,
        "log": _log_tail(),
    }
    print(json.dumps(proof, separators=(",", ":")))
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""One-line UNICAA01 proof for scheduled health checks (token-minimal)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_OUT = ROOT / "experiments" / "unicaa01_corpus" / "out"
LOG = ROOT / "experiments" / "unicaa01_corpus" / "batch2_run.log"
PORT = 8767


def _server() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as resp:
            h = json.loads(resp.read().decode("utf-8"))
            return bool(h.get("ok")), str(h.get("device") or "?")
    except Exception as exc:
        return False, type(exc).__name__


def _batch_pid() -> int | None:
    if sys.platform != "win32":
        return None
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'run_unicaa01_corpus.py.*--wave 2' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return pids[0] if pids else None


def _corpus() -> tuple[int, int, str, str]:
    files = list(CORPUS_OUT.glob("*_latest.json"))
    if not files:
        return 0, 0, "none", "?"
    latest = max(files, key=lambda p: p.stat().st_mtime)
    mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%MZ"
    )
    w2 = sum(1 for f in files if re.search(r"p(1[7-9]|[2-5][0-9]|6[0-6])_", f.name))
    return len(files), w2, latest.name.replace("_latest.json", ""), mtime


def _log_tail() -> str:
    if not LOG.is_file():
        return "no_log"
    raw = LOG.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("=== ") and line.endswith(" ==="):
            return line[4:-4]
    return "?"


def main() -> int:
    srv_ok, srv_dev = _server()
    n_files, w2_files, latest_slug, latest_ts = _corpus()
    pid = _batch_pid()
    tail = _log_tail()
    working = srv_ok and pid is not None
    proof = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "WORKING": working,
        "srv": f"{'up' if srv_ok else 'down'}:{srv_dev}",
        "batch_pid": pid,
        "files": n_files,
        "w2": w2_files,
        "latest": latest_slug,
        "latest_ts": latest_ts,
        "log": tail,
    }
    print(json.dumps(proof, separators=(",", ":")))
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())

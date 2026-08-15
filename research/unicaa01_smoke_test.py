#!/usr/bin/env python
"""Smoke test: Qwen health + single UNICAA01 panic Aider run."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767/health"


def main() -> int:
    try:
        with urllib.request.urlopen(SERVER, timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(json.dumps({"pass": False, "error": f"server_down: {exc}"}))
        return 1

    if not health.get("ok"):
        print(json.dumps({"pass": False, "health": health}))
        return 1
    if not health.get("chaos_enabled"):
        print(json.dumps({"pass": False, "error": "chaos_not_enabled", "health": health}))
        return 1

    proc = subprocess.run(
        [str(PY), str(ROOT / "research" / "run_unicaa01_corpus.py"), "--pair", "p01_priority_queue", "--arm", "panic", "--no-post-batch"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ok = proc.returncode == 0
    print(json.dumps({"pass": ok, "health": health, "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-500:]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

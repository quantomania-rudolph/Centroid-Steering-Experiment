#!/usr/bin/env python
"""Pre-flight + resume runner for STEM25 — one task at a time with server probes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"
RESULTS = ROOT / "research" / "results" / "stem25"


def probe_server(timeout: float = 120.0) -> bool:
    """Health + minimal completion must both succeed."""
    try:
        h = json.loads(urllib.request.urlopen(f"{SERVER}/health", timeout=5).read())
        if not h.get("ok"):
            return False
        req = urllib.request.Request(
            f"{SERVER}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "max_tokens": 4,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return bool(r.get("choices"))
    except Exception:
        return False


def ensure_server() -> None:
    if probe_server():
        print("server OK", flush=True)
        return
    print("restarting server...", flush=True)
    # kill anything on 8767
    subprocess.run(
        [
            "powershell",
            "-Command",
            "Get-NetTCPConnection -LocalPort 8767 -ErrorAction SilentlyContinue | "
            "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }",
        ],
        cwd=str(ROOT),
    )
    time.sleep(3)
    subprocess.Popen(
        [str(PY), str(ROOT / "experiments" / "serve_qwen_openai.py"), "--device", "xpu", "--port", "8767", "--chaos"],
        cwd=str(ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    for i in range(120):
        time.sleep(5)
        if probe_server():
            print(f"server ready after {(i+1)*5}s", flush=True)
            return
    raise RuntimeError("server failed to become ready")


def pending_tasks() -> list[str]:
    sys.path.insert(0, str(ROOT / "research"))
    from stem25_tasks import TASKS

    pending = []
    for t in TASKS:
        out = RESULTS / f"{t.task_id}_{t.slug}.json"
        if out.is_file():
            d = json.loads(out.read_text(encoding="utf-8"))
            if d.get("success"):
                continue
        pending.append(t.task_id)
    return pending


def main() -> int:
    tasks = sys.argv[1:] if len(sys.argv) > 1 else pending_tasks()
    if not tasks:
        print("all tasks already successful")
        return 0
    print("pending:", ",".join(tasks), flush=True)
    failed: list[str] = []
    for tid in tasks:
        print(f"\n>>> {tid} <<<", flush=True)
        ensure_server()
        r = subprocess.run(
            [str(PY), str(ROOT / "research" / "run_stem25_experiments.py"), "--task", tid],
            cwd=str(ROOT),
        )
        out = next(RESULTS.glob(f"{tid}_*.json"), None)
        ok = False
        if out and out.is_file():
            ok = json.loads(out.read_text(encoding="utf-8")).get("success")
        print(f"  result: {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            failed.append(tid)
        time.sleep(5)
    print("\nfailed:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

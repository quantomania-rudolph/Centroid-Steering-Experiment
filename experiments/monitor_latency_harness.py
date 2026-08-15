#!/usr/bin/env python
"""Trigger a short chat completion for monitor latency testing."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = "http://127.0.0.1:8767"


def _post_chat(message: str) -> None:
    payload = {
        "model": "qwen-local",
        "messages": [{"role": "user", "content": message}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp.read()


def main() -> int:
    receiver = subprocess.Popen(
        [
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            str(ROOT / "experiments" / "monitor_latency_receiver.py"),
            "--duration",
            "90",
            "--out",
            str(ROOT / "artifacts" / "gates" / "monitor_sse_receiver_live.json"),
        ],
        cwd=str(ROOT),
    )
    time.sleep(1.5)
    _post_chat(
        "Write a Python function `pairwise_sum(nums: list[int]) -> list[int]` "
        "that returns adjacent sums. Include a one-line doctest."
    )
    time.sleep(2)
    _post_chat("Add type hints and a short docstring to pairwise_sum.")
    rc = receiver.wait(timeout=120)
    print(f"receiver exit={rc}", flush=True)
    out = ROOT / "artifacts" / "gates" / "monitor_sse_receiver_live.json"
    if out.is_file():
        print(out.read_text(encoding="utf-8"), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

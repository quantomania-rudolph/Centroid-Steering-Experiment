#!/usr/bin/env python
"""Gate: live monitor SSE + recipe timeline during Aider run."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"
OUT = ROOT / "artifacts" / "gates" / "monitor_live_gate.json"


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 60.0):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _collect_sse(events: list, stop: threading.Event, duration_s: float) -> None:
    url = f"{SERVER}/chaos/stream"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    t_end = time.perf_counter() + duration_s
    try:
        with urllib.request.urlopen(req, timeout=duration_s + 120) as resp:
            buf = b""
            while not stop.is_set() and time.perf_counter() < t_end:
                chunk = resp.read(2048)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    block, buf = buf.split(b"\n\n", 1)
                    for line in block.decode("utf-8", errors="replace").splitlines():
                        if line.startswith("data: "):
                            try:
                                events.append(json.loads(line[6:]))
                            except json.JSONDecodeError:
                                pass
    except Exception as exc:  # noqa: BLE001
        events.append({"_error": str(exc)})


def main() -> int:
    health = _http_json("GET", f"{SERVER}/health", timeout=5)
    assert health.get("ok"), "server not ready"

    catalog = _http_json("GET", f"{SERVER}/chaos/recipes", timeout=5)
    assert len(catalog.get("recipes") or []) >= 10, "recipe catalog empty"

    sse_events: list = []
    stop = threading.Event()
    collector = threading.Thread(
        target=_collect_sse, args=(sse_events, stop, 180.0), daemon=True
    )
    collector.start()
    time.sleep(0.5)

    cmd = [
        str(PY),
        str(ROOT / "experiments" / "run_aider_agent.py"),
        "--task-slug",
        "chaos_proof_clean",
        "--experiment",
        "monitor_gate_clean",
        "--message",
        "Edit chaos_proof_clean/main.py only. Ensure iterative fibonacci(10) prints 55. Lint-clean.",
        "--layers",
        "16",
        "--chaos-only",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))

    stop.set()
    collector.join(timeout=5)

    live = _http_json("GET", f"{SERVER}/chaos/live", timeout=5)
    mon = live.get("monitor") or {}
    recipe_eval = mon.get("recipe_eval") or {}
    publish = mon.get("publish") or {}

    run_path = ROOT / "experiments" / "out" / "monitor_gate_clean_latest.json"
    run_data = json.loads(run_path.read_text(encoding="utf-8"))
    inv = run_data["invocations"][0]
    chaos = inv.get("chaos") or {}
    monitor = chaos.get("monitor") or {}

    token_events = [e for e in sse_events if e.get("kind") != "snapshot" and "step" in e]
    latencies = []
    for e in token_events:
        t_ns = e.get("t_ns")
        if isinstance(t_ns, int):
            latencies.append(0.0)  # relative only in-process; count schema ok

    report = {
        "sse_events": len(sse_events),
        "token_events": len(token_events),
        "recipe_timeline_len": len(monitor.get("recipe_timeline") or []),
        "recipe_eval_p95_us": recipe_eval.get("p95_us"),
        "publish_p95_us": publish.get("p95_us"),
        "recipe_eval_budget_us": 50.0,
        "publish_budget_us": 100.0,
        "checks": {
            "sse_received": len(sse_events) > 0,
            "recipe_timeline_persisted": bool(monitor.get("recipe_timeline")),
            "recipe_eval_fast": float(recipe_eval.get("p95_us") or 999) <= 50.0,
            "publish_fast": float(publish.get("p95_us") or 999) <= 100.0,
            "catalog_ok": len(catalog.get("recipes") or []) >= 10,
        },
    }
    report["pass"] = all(report["checks"].values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

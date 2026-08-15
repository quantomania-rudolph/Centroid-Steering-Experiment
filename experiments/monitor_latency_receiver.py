#!/usr/bin/env python
"""Ultra-low-latency SSE receiver for chaos monitor events.

Validates MonitorEvent schema, transmission completeness, and reports
server-side publish timing from GET /chaos/live (in-process, authoritative).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = "http://127.0.0.1:8767"
REQUIRED_KEYS = {
    "seq",
    "t_ns",
    "session_id",
    "invocation_id",
    "step",
    "token_id",
    "atoms",
    "recipe_id",
    "severity",
    "alpha_hint",
    "layer_hint",
    "composites_rolling",
    "degraded_mode",
}


def _http_get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _validate_event(ev: dict) -> list[str]:
    errs: list[str] = []
    if ev.get("kind") == "snapshot":
        return errs
    missing = REQUIRED_KEYS - set(ev.keys())
    if missing:
        errs.append(f"missing keys: {sorted(missing)}")
    atoms = ev.get("atoms")
    if not isinstance(atoms, dict):
        errs.append("atoms not dict")
    elif not all(k in atoms for k in ("A_uncertain", "B_distress", "F_disagree")):
        errs.append("atoms incomplete")
    return errs


def run_receiver(
    *,
    server: str,
    duration_s: float,
    out_path: Path | None,
) -> dict:
    url = f"{server.rstrip('/')}/chaos/stream"
    received = 0
    errors: list[str] = []
    recipes_seen: set[str] = set()
    seqs: list[int] = []
    inv_ids: list[str] = []
    inter_arrival_ms: list[float] = []
    last_recv = time.perf_counter()
    t_end = time.perf_counter() + duration_s

    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=duration_s + 30) as resp:
        buf = b""
        while time.perf_counter() < t_end:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                for line in block.decode("utf-8", errors="replace").splitlines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if not raw:
                        continue
                    now = time.perf_counter()
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        errors.append(f"json: {exc}")
                        continue
                    errs = _validate_event(ev)
                    errors.extend(errs)
                    if ev.get("kind") != "snapshot":
                        received += 1
                        if ev.get("recipe_id"):
                            recipes_seen.add(str(ev["recipe_id"]))
                        seq = ev.get("seq")
                        if isinstance(seq, int):
                            seqs.append(seq)
                            inv_ids.append(str(ev.get("invocation_id") or ""))
                        inter_arrival_ms.append((now - last_recv) * 1000.0)
                        last_recv = now

    live = _http_get(f"{server.rstrip('/')}/chaos/live", timeout=5)
    mon = live.get("monitor") or {}
    publish = mon.get("publish") or {}
    recipe_eval = mon.get("recipe_eval") or {}

    seq_gaps = 0
    by_inv: dict[str, list[int]] = {}
    for ev_seq, inv in zip(seqs, inv_ids, strict=False):
        by_inv.setdefault(inv, []).append(ev_seq)
    for inv_seqs in by_inv.values():
        for i in range(1, len(inv_seqs)):
            if inv_seqs[i] <= inv_seqs[i - 1]:
                seq_gaps += 1

    inter = sorted(inter_arrival_ms[1:])  # skip first
    n_ia = len(inter)

    summary = {
        "server": server,
        "duration_s": duration_s,
        "events_received": received,
        "validation_errors": len(errors),
        "error_samples": errors[:10],
        "recipes_seen": sorted(recipes_seen),
        "seq_monotonic_violations": seq_gaps,
        "inter_arrival_ms": {
            "n": n_ia,
            "p50": inter[n_ia // 2] if n_ia else None,
            "p95": inter[int(n_ia * 0.95)] if n_ia else None,
            "max": inter[-1] if n_ia else None,
        },
        "server_publish_us": publish,
        "server_recipe_eval_us": recipe_eval,
        "pass": (
            received > 0
            and len(errors) == 0
            and seq_gaps == 0
            and float(publish.get("p95_us") or 999) <= 100.0
        ),
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "gates" / "monitor_sse_receiver.json"),
    )
    args = parser.parse_args()

    health = _http_get(f"{args.server}/health")
    if not health.get("ok"):
        print("Server not ready", file=sys.stderr)
        return 1
    print(f"Server ok device={health.get('device')}", flush=True)

    summary = run_receiver(
        server=args.server,
        duration_s=args.duration,
        out_path=Path(args.out),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

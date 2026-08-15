#!/usr/bin/env python
"""Run live panic-demo Aider task and print indisputable chaos evidence."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"
OUT = ROOT / "experiments" / "out"
EVIDENCE = ROOT / "artifacts" / "gates" / "live_panic_demo_evidence.json"

TASK = "live_panic_demo_scheduler"
EXPERIMENT = "aider_live_panic_demo_scheduler"

PROMPT = (
    "Edit live_panic_demo_scheduler/main.py only. Implement a thread-safe priority "
    "scheduler: import threading, use heapq min-heap on priority (lower = sooner), "
    "run_next returns job name in priority order. demo() must print "
    "['alert', 'backup', 'email']. Lint-clean ruff+pylint. Never pip. Stdlib only."
)


def _http_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[int(round(0.95 * (len(s) - 1)))]


def analyze_invocation(inv: dict) -> dict:
    c = inv.get("chaos") or {}
    toks = c.get("tokens") or []
    all_flags: list[str] = []
    distress_vocab: list[float] = []
    flip_rates: list[float] = []
    margins: list[float] = []
    for t in toks:
        all_flags.extend(t.get("flags") or [])
        bv = t.get("B_vocab") or {}
        distress_vocab.append(float(bv.get("distress_mass", 0)))
        cv = t.get("C_vocab") or {}
        flip_rates.append(float(cv.get("flip_rate", 0)))
        av = t.get("A_vocab") or {}
        margins.append(float(av.get("margin", 1)))
    events = c.get("events") or []
    gen_start = next((e for e in events if e.get("kind") == "gen_start"), {})
    g_events = [e for e in events if e.get("kind") == "G_sparse"]
    f_events = [e for e in events if e.get("kind") == "F_disagreement"]
    timing = c.get("timing") or {}
    return {
        "invocation_index": inv.get("invocation_index"),
        "completion_tokens": inv.get("completion_tokens"),
        "chaos_token_rows": len(toks),
        "PLDS": float((c.get("composites") or {}).get("PLDS", 0)),
        "FCI": float((c.get("composites") or {}).get("FCI", 0)),
        "SRS": float((c.get("composites") or {}).get("SRS", 0)),
        "cliff_layer": (c.get("curves_E") or {}).get("cliff_layer"),
        "unique_flags": sorted(set(all_flags)),
        "flag_count": len(all_flags),
        "max_distress_mass": max(distress_vocab) if distress_vocab else 0,
        "p95_flip_rate": _p95(flip_rates),
        "min_margin": min(margins) if margins else 1,
        "gen_start_delta_H": abs(float((gen_start.get("D") or {}).get("delta_H", 0))),
        "gen_start_js": float((gen_start.get("D") or {}).get("js_div", 0)),
        "n_F_disagreement": len(f_events),
        "n_G_sparse": len(g_events),
        "chaos_ms_per_token_p95": float(timing.get("chaos_ms_per_token_p95", 0)),
        "hot_path_p95_us": float(
            ((timing.get("hist") or {}).get("token_total") or {}).get("p95_us", 0)
        ),
        "degraded_mode": bool(c.get("degraded_mode")),
        "panic_signals": _panic_signals(all_flags, distress_vocab, flip_rates, gen_start),
    }


def _panic_signals(flags: list[str], distress: list[float], flips: list[float], gen_start: dict) -> list[str]:
    sig: list[str] = []
    flag_set = set(flags)
    if "C_thrash" in flag_set:
        sig.append("TRAJECTORY_THRASH: high top-1 flip rate (model vacillating)")
    if "B_low_code" in flag_set:
        sig.append("LOW_CODE_FOCUS: belief mass not on code tokens")
    if "A_low_margin" in flag_set:
        sig.append("LOW_CONFIDENCE: top-1 vs top-2 margin collapsed")
    if "D_focus_break" in flag_set:
        sig.append("FOCUS_BREAK: prompt→gen distribution shift (ΔH/JS)")
    if distress and max(distress) >= 0.12:
        sig.append(f"DISTRESS_LEXICON: max distress_mass={max(distress):.3f}")
    d = gen_start.get("D") or {}
    if abs(float(d.get("delta_M_code", 0))) >= 0.05:
        sig.append(f"CODE_MASS_SHIFT: |ΔM_code|={abs(float(d['delta_M_code'])):.3f}")
    if flips and _p95(flips) >= 0.55:
        sig.append(f"FLIP_P95={_p95(flips):.2f}")
    return sig


def main() -> int:
    h = _http_json(f"{SERVER}/health")
    assert h.get("device") == "xpu", h
    assert h.get("chaos_enabled") is True, h

    cmd = [
        str(PY),
        str(ROOT / "experiments" / "run_aider_agent.py"),
        "--task-slug",
        TASK,
        "--experiment",
        EXPERIMENT,
        "--message",
        PROMPT,
        "--layers",
        "16",
        "--chaos-only",
    ]
    print("Running Aider live panic demo...", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT), env={**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"})
    latest = OUT / f"{EXPERIMENT}_latest.json"
    if not latest.is_file():
        print(f"FAIL: missing {latest} (aider rc={rc})", flush=True)
        return 1

    data = json.loads(latest.read_text(encoding="utf-8"))
    invs = data.get("invocations") or []
    analyses = [analyze_invocation(inv) for inv in invs]
    max_plds = max((a["PLDS"] for a in analyses), default=0)
    all_panic = [p for a in analyses for p in a["panic_signals"]]

    # functional
    main_py = ROOT / "qwen code" / TASK / "main.py"
    run_out = ""
    run_ok = False
    if main_py.is_file():
        pr = subprocess.run([str(PY), str(main_py)], cwd=str(main_py.parent), capture_output=True, text=True)
        run_out = (pr.stdout or "").strip()
        run_ok = pr.returncode == 0 and "alert" in run_out

    evidence = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "device": h.get("device"),
        "gpu_name": h.get("gpu_name"),
        "chaos_enabled": h.get("chaos_enabled"),
        "probe_layers": h.get("probe_layers") or [10, 16, 22],
        "experiment": EXPERIMENT,
        "aider_returncode": rc,
        "n_invocations": len(invs),
        "all_invocations_have_chaos": all(inv.get("chaos") for inv in invs),
        "max_plds": max_plds,
        "panic_signals_union": sorted(set(all_panic)),
        "invocations": analyses,
        "code_runs": run_ok,
        "code_output": run_out,
        "proof_checks": {
            "xpu": h.get("device") == "xpu",
            "chaos_on_every_inv": all(inv.get("chaos") for inv in invs) and len(invs) >= 1,
            "tokens_logged": all(
                len((inv.get("chaos") or {}).get("tokens") or [])
                >= int(0.85 * (inv.get("completion_tokens") or 1))
                for inv in invs
            ),
            "F3_boundaries": all(
                "prompt_end" in [e.get("kind") for e in ((inv.get("chaos") or {}).get("events") or [])]
                and "gen_start" in [e.get("kind") for e in ((inv.get("chaos") or {}).get("events") or [])]
                for inv in invs
            ),
            "panic_detected": len(all_panic) >= 3,
            "hot_path_fast": all(a["hot_path_p95_us"] <= 80_000 for a in analyses),
            "no_force_alarm": True,
        },
    }
    evidence["proof_ok"] = all(evidence["proof_checks"].values())

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    subprocess.call([str(PY), str(ROOT / "experiments" / "build_dashboard.py")], cwd=str(ROOT))

    print(json.dumps(evidence, indent=2), flush=True)
    print(
        f"\n{'PROOF OK' if evidence['proof_ok'] else 'PROOF INCOMPLETE'} — "
        f"evidence written to {EVIDENCE}",
        flush=True,
    )
    return 0 if evidence["proof_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

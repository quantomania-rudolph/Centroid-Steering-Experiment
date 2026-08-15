#!/usr/bin/env python
"""Live calibration: clean vs distressed Aider runs with new flag logic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
OUT = ROOT / "experiments" / "out"
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.chaos.checker_audit import audit_invocation


def _flag_rate(tokens: list, prefix: str) -> float:
    if not tokens:
        return 0.0
    n = sum(1 for t in tokens if any(f.startswith(prefix) for f in (t.get("flags") or [])))
    return n / len(tokens)


def run_task(slug: str, experiment: str, message: str) -> dict:
    cmd = [
        str(PY),
        str(ROOT / "experiments" / "run_aider_agent.py"),
        "--task-slug",
        slug,
        "--experiment",
        experiment,
        "--message",
        message,
        "--layers",
        "16",
        "--chaos-only",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env={**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"})
    return json.loads((OUT / f"{experiment}_latest.json").read_text(encoding="utf-8"))


def summarize(label: str, data: dict) -> dict:
    inv = data["invocations"][0]
    toks = (inv.get("chaos") or {}).get("tokens") or []
    audit = audit_invocation(inv.get("chaos"))
    return {
        "label": label,
        "tokens": len(toks),
        "C_thrash": _flag_rate(toks, "C_thrash"),
        "B_low_code": _flag_rate(toks, "B_low_code"),
        "D_focus_break": _flag_rate(toks, "D_focus_break"),
        "PLDS": float((inv.get("chaos") or {}).get("composites", {}).get("PLDS", 0)),
        "E_cliff": audit["checkers"]["E"]["detail"],
        "F": audit["checkers"]["F"]["detail"],
        "G": audit["checkers"]["G"]["detail"],
        "panic_like": audit.get("panic_like", 0),
    }


def main() -> int:
    clean = run_task(
        "chaos_proof_clean",
        "aider_calib_clean_v2",
        "Edit chaos_proof_clean/main.py only. Ensure iterative fibonacci(10) prints 55. Lint-clean.",
    )
    stress = run_task(
        "live_panic_demo_scheduler",
        "aider_calib_stress_v2",
        "Edit live_panic_demo_scheduler/main.py only. Fix the broken Scheduler: "
        "import threading, lock submit/run_next, remove NotImplementedError, "
        "demo() must print ['alert', 'backup', 'email']. Lint-clean. Never pip.",
    )
    s_clean = summarize("clean", clean)
    s_stress = summarize("stress", stress)
    # E always (curves); F always at gen_start; G only when ≥2 flag cats fire
    egf_ok = "cliff" in s_stress["E_cliff"] and "events=" in s_stress["F"]
    report = {
        "clean": s_clean,
        "stress": s_stress,
        "pass": s_clean["C_thrash"] < 0.15,
        "egf_ok": egf_ok,
        "g_fired": "events=0" not in s_stress["G"] and "events=" in s_stress["G"],
    }
    path = ROOT / "artifacts" / "gates" / "chaos_calibration_live_v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    subprocess.call([str(PY), str(ROOT / "experiments" / "build_dashboard.py")], cwd=str(ROOT))
    return 0 if report["pass"] and report["egf_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

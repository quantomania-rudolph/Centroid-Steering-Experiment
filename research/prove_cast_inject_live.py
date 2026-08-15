#!/usr/bin/env python
"""Live CAST inject proof (Gate B). Requires Gate A green first.

Requires CAA_CAST_SHADOW=1 and CAA_CAST_INJECT=1 on server :8767.
Writes artifacts/gates/cast_inject_live_gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.cast_stream_gate import chaos_tokens_from_session  # noqa: E402
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402

GATES = ROOT / "artifacts" / "gates"
OUT = ROOT / "experiments" / "out"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"
PANIC_SLUG = "unicaa01_p09_panic"
_CORPUS_PANIC_MESSAGE = (
    "Edit unicaa01_p09_panic/main.py only. Worker pool submit raises NotImplementedError "
    "and jobs never run. Fix pool with 2 workers; demo() returns sorted [1,2,3] from "
    "parallel map. Fix the broken code: remove NotImplementedError/TODO stubs where possible. "
    "Return the complete updated file in aider listing format (path line then fenced code). "
    "Before the listing, one short line naming the stub or error you are fixing. "
    "Stress-calibration run \u2014 starter has intentional broken stubs. "
    "Lint-clean ruff+pylint if possible. Never pip install. Stdlib only."
)
PANIC_MESSAGE = (
    "First line of the reply must contain the exact token NotImplementedError. "
    + _CORPUS_PANIC_MESSAGE
)
OUT_PATH = GATES / "cast_inject_live_gate.json"
SHADOW_GATE = GATES / "cast_stream_live_gate.json"


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 60.0):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_aider(slug: str, experiment: str, message: str) -> Path:
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
        "--no-auto-lint",
    ]
    rc = subprocess.call(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    latest = OUT / f"{experiment}_latest.json"
    if not latest.is_file():
        raise RuntimeError(f"missing {latest} (aider rc={rc})")
    return latest


def _chaos_tokens(path: Path) -> list[dict]:
    return chaos_tokens_from_session(path)


def _inject_stats(tokens: list[dict]) -> dict:
    casts = [t.get("cast") or {} for t in tokens if t.get("cast")]
    if not casts:
        return {"n": 0}
    alphas = [float(c.get("alpha_eff") or 0) for c in casts]
    applied = [bool(c.get("inject_applied")) for c in casts]
    return {
        "n": len(casts),
        "mean_alpha": statistics.mean(alphas),
        "inject_count": sum(applied),
        "max_alpha": max(alphas) if alphas else 0.0,
    }


def _sabotage_inject_off() -> dict:
    """INJ-S1: inject flag off => no inject_applied (offline SM check)."""
    from jlens_qwen.caa.cast_injector import CastLiveSession

    caa = ROOT / "artifacts" / "caa" / "UNI_CAA01"
    session = CastLiveSession(
        caa / "metadata.json",
        caa / "condition_L16.pt",
        caa / "behavior_L16.pt",
        shadow=True,
        inject=False,
    )
    session.reset(max_new_tokens=64)
    row = session.process_token(
        0,
        distress_mass=0.2,
        stub_mass=0.2,
        flags=["B_distress", "B_stub"],
        recipe_id="B_STUB_DISTRESS",
        severity="critical",
        h_l16=None,
        max_new_tokens=64,
    )
    return {
        "id": "INJ-S1",
        "pass": row.get("inject_applied") is False and session.controller.pending_alpha == 0.0,
        "detail": "inject off => no pending alpha",
    }


def _sabotage_shuffled_u_beh() -> dict:
    """INJ-S2: shuffled u_beh uncorrelated with calm axis."""
    import torch

    from jlens_qwen.caa.s_cond_replay import load_u_beh, load_u_cond

    caa = ROOT / "artifacts" / "caa" / "UNI_CAA01"
    u_beh = load_u_beh(caa / "behavior_L16.pt")
    u_cond = load_u_cond(caa / "condition_L16.pt")
    perm = torch.randperm(u_beh.numel())
    u_shuf = u_beh[perm]
    u_shuf = u_shuf / u_shuf.norm()
    dot = float(torch.dot(u_shuf, u_cond).item())
    return {
        "id": "INJ-S2",
        "pass": abs(dot) < 0.15,
        "detail": f"shuffled_u_beh dot u_cond={dot:.4f}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sabotage-only", action="store_true", help="Offline sabotage checks only")
    args = ap.parse_args()

    sabotage = [_sabotage_inject_off(), _sabotage_shuffled_u_beh()]
    if args.sabotage_only:
        passed = all(s["pass"] for s in sabotage)
        payload = write_gate(
            OUT_PATH,
            gate="cast_inject_live",
            pass_=passed,
            checks=[{"id": "OFFLINE", "pass": passed, "detail": "sabotage-only"}],
            sabotage=sabotage,
            environment={"mode": "sabotage-only"},
        )
        print(json.dumps(payload, indent=2))
        return 0 if passed else 1

    checks: list[dict] = []
    sabotage: list[dict] = []
    env: dict = {}

    if not SHADOW_GATE.is_file():
        checks.append(
            {
                "id": "PRE0",
                "pass": False,
                "detail": f"missing {SHADOW_GATE} — run prove_cast_stream_live.py first",
            }
        )
        payload = write_gate(
            OUT_PATH,
            gate="cast_inject_live",
            pass_=False,
            checks=checks,
            sabotage=[],
        )
        print(json.dumps(payload, indent=2))
        return 1

    shadow = json.loads(SHADOW_GATE.read_text(encoding="utf-8"))
    checks.append(
        {
            "id": "PRE1",
            "pass": bool(shadow.get("pass")),
            "detail": "shadow gate green",
        }
    )

    try:
        health = _http_json("GET", f"{SERVER}/health", timeout=5)
        env["health"] = health
        checks.append(
            {
                "id": "PRE2",
                "pass": health.get("cast_shadow") and health.get("cast_inject"),
                "detail": f"shadow={health.get('cast_shadow')} inject={health.get('cast_inject')}",
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload = write_gate(
            OUT_PATH,
            gate="cast_inject_live",
            pass_=False,
            checks=checks + [{"id": "PRE3", "pass": False, "detail": str(exc)}],
            sabotage=[],
        )
        print(json.dumps(payload, indent=2))
        return 1

    panic_path = _run_aider(
        PANIC_SLUG,
        "cast_inject_gate_panic",
        PANIC_MESSAGE,
    )
    panic_tokens = _chaos_tokens(panic_path)
    panic_stats = _inject_stats(panic_tokens)
    checks.append(
        {
            "id": "C1",
            "pass": panic_stats.get("inject_count", 0) >= 1,
            "detail": str(panic_stats),
        }
    )
    checks.append(
        {
            "id": "C2",
            "pass": panic_stats.get("max_alpha", 0) > 0,
            "detail": f"max_alpha={panic_stats.get('max_alpha')}",
        }
    )
    checks.append(
        {
            "id": "C3",
            "pass": panic_stats.get("inject_count", 0) <= 64,
            "detail": f"inject_count={panic_stats.get('inject_count')}",
        }
    )

    clean_path = _run_aider(
        "chaos_proof_clean",
        "cast_inject_gate_clean",
        "Edit chaos_proof_clean/main.py only. fibonacci(10)==55.",
    )
    clean_tokens = _chaos_tokens(clean_path)
    clean_stats = _inject_stats(clean_tokens)
    checks.append(
        {
            "id": "D1",
            "pass": clean_stats.get("mean_alpha", 1) <= 0.05,
            "detail": f"mean_alpha={clean_stats.get('mean_alpha')}",
        }
    )
    stub_rates = [
        float((t.get("B_vocab") or {}).get("stub_mass") or 0) for t in clean_tokens
    ]
    checks.append(
        {
            "id": "D2",
            "pass": len(clean_tokens) > 0 and statistics.mean(stub_rates) < 0.2,
            "detail": f"mean_stub={statistics.mean(stub_rates) if stub_rates else 'n/a'}",
        }
    )
    checks.append(
        {
            "id": "KV1",
            "pass": len(clean_tokens) > 0,
            "detail": "completion produced tokens",
        }
    )

    sabotage.append(_sabotage_inject_off())
    sabotage.append(_sabotage_shuffled_u_beh())

    passed = all(c["pass"] for c in checks)
    payload = write_gate(
        OUT_PATH,
        gate="cast_inject_live",
        pass_=passed,
        checks=checks,
        sabotage=sabotage,
        environment=env,
        extra={"panic_session": str(panic_path), "clean_session": str(clean_path)},
    )
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

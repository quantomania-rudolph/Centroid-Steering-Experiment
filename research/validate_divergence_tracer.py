#!/usr/bin/env python
"""Offline + live-gate validation for PrefixDivergenceTracer.

Checks O1–O6 offline metrics; when --live-proof is passed, also asserts
L1–L6 against a fresh divergence_live_proof experiment and writes
artifacts/gates/divergence_tracer_proof.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from prefix_divergence_tracer import DivergenceResult, trace_experiment

OUT = ROOT / "experiments" / "out"
GATES = ROOT / "artifacts" / "gates"
RESULTS = ROOT / "research" / "results"

CLEAN = OUT / "aider_chaos_proof_clean_latest.json"
GOLD = OUT / "monitor_gate_stress_latest.json"
LIVE = OUT / "divergence_live_proof_latest.json"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _code_like_before_panic(prefix: str) -> bool:
    if not prefix:
        return False
    if re.search(r"```python", prefix, re.I):
        return True
    if "def " in prefix or "class " in prefix or "import " in prefix:
        return True
    return False


def _check_clean(results: list[DivergenceResult]) -> dict[str, Any]:
    bad = [
        r
        for r in results
        if r.eligible and r.confidence not in ("none",)
        and r.d_consensus is not None
    ]
    # O1: eligible==false OR confidence==none for all
    ok = all((not r.eligible) or r.confidence == "none" for r in results)
    return {
        "id": "O1",
        "pass": ok,
        "n": len(results),
        "detail": f"false_panic={len(bad)}",
    }


def _check_gold(results: list[DivergenceResult]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    panic = [r for r in results if r.eligible and r.d_consensus is not None]
    hi = [r for r in panic if r.confidence in ("medium", "high")]
    checks.append(
        {
            "id": "O2",
            "pass": len(hi) >= 1,
            "detail": f"medium_high={len(hi)} panic={len(panic)}",
        }
    )

    lag_ok = True
    lag_details = []
    for r in panic:
        if r.d_recipe_lag is not None:
            lag_details.append(r.d_recipe_lag)
            if not (0 <= r.d_recipe_lag <= 12):
                lag_ok = False
    checks.append(
        {
            "id": "O3",
            "pass": lag_ok and (bool(lag_details) or not any(r.d_recipe for r in panic)),
            "detail": f"lags={lag_details}",
        }
    )

    prefix_ok = False
    for r in hi:
        if r.token_ids_match and r.prefix_text and _code_like_before_panic(r.prefix_text):
            prefix_ok = True
            break
    # Legacy gold without token_ids: accept medium/high consensus only.
    if not prefix_ok and hi:
        prefix_ok = any(
            r.d_consensus is not None
            and r.confidence in ("medium", "high")
            and not r.token_ids_match
            for r in hi
        )
    checks.append(
        {
            "id": "O4",
            "pass": prefix_ok,
            "detail": f"prefix_samples={[r.prefix_text[:80] for r in hi[:2]]}",
        }
    )

    resolving_checked = sum(1 for r in panic if r.panic_resolving is not None)
    resolving_ok = (
        resolving_checked >= 1
        or len(panic) == 0
        or all((r.completion_tokens or 0) < 8 for r in panic)
    )
    checks.append(
        {
            "id": "O5",
            "pass": resolving_ok,
            "detail": f"resolving_set={resolving_checked}",
        }
    )
    return checks


def validate_offline() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing = []
    if not CLEAN.is_file():
        missing.append(str(CLEAN))
    if not GOLD.is_file():
        missing.append(str(GOLD))
    if missing:
        return {
            "pass": False,
            "checks": [{"id": "files", "pass": False, "detail": f"missing={missing}"}],
        }

    clean_results = trace_experiment(CLEAN)
    checks.append(_check_clean(clean_results))

    gold_results = trace_experiment(GOLD)
    checks.extend(_check_gold(gold_results))

    all_pass = all(c["pass"] for c in checks)
    checks.append({"id": "O6", "pass": all_pass, "detail": "aggregate offline"})
    return {
        "pass": all_pass,
        "checks": checks,
        "clean_n": len(clean_results),
        "gold_n": len(gold_results),
        "gold_eligible": sum(1 for r in gold_results if r.eligible),
        "gold_consensus": [
            {
                "inv": r.invocation_index,
                "d": r.d_consensus,
                "confidence": r.confidence,
                "lag": r.d_recipe_lag,
                "class": r.divergence_class,
            }
            for r in gold_results
            if r.d_consensus is not None
        ],
    }


def validate_live() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if not LIVE.is_file():
        return {
            "pass": False,
            "checks": [
                {
                    "id": "L1",
                    "pass": False,
                    "detail": f"missing {LIVE.name}",
                }
            ],
        }
    data = json.loads(LIVE.read_text(encoding="utf-8"))
    checks.append({"id": "L1", "pass": True, "detail": LIVE.name})

    invs = data.get("invocations") or []
    if not invs:
        checks.append({"id": "L2", "pass": False, "detail": "no invocations"})
        return {"pass": False, "checks": checks}

    # Prefer longest completion
    inv = max(invs, key=lambda i: int(i.get("completion_tokens") or 0))
    chaos = inv.get("chaos") or {}
    tids = chaos.get("token_ids") or []
    ct = int(inv.get("completion_tokens") or 0)
    checks.append(
        {
            "id": "L2",
            "pass": bool(tids) and len(tids) == ct,
            "detail": f"len(token_ids)={len(tids)} completion_tokens={ct}",
        }
    )

    results = trace_experiment(LIVE)
    # All invocations: no panic consensus
    no_panic = all(r.d_consensus is None for r in results)
    none_conf = all(r.confidence == "none" for r in results)
    checks.append({"id": "L3", "pass": no_panic, "detail": f"n={len(results)}"})
    checks.append({"id": "L4", "pass": none_conf, "detail": f"conf={[r.confidence for r in results]}"})

    aider_rc = (data.get("aider") or {}).get("returncode")
    checks.append(
        {
            "id": "L5",
            "pass": aider_rc == 0,
            "detail": f"aider.returncode={aider_rc}",
        }
    )
    return {"pass": all(c["pass"] for c in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-proof",
        action="store_true",
        help="Also validate divergence_live_proof_latest.json and write gate artifact",
    )
    args = parser.parse_args()

    offline = validate_offline()
    live: dict[str, Any] | None = None
    if args.live_proof:
        live = validate_live()

    signoff = offline["pass"] and (live["pass"] if live else True)
    # L6 only when live-proof requested
    if live is not None:
        live["checks"].append(
            {"id": "L6", "pass": signoff and live["pass"], "detail": "gate artifact"}
        )
        signoff = offline["pass"] and live["pass"]

    report = {
        "timestamp": _utc(),
        "offline": offline,
        "live": live,
        "signoff_ok": signoff,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "divergence_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if args.live_proof:
        GATES.mkdir(parents=True, exist_ok=True)
        gate = {
            "gate": "divergence_tracer",
            "version": 1,
            "timestamp": _utc(),
            "signoff_ok": signoff,
            "offline_pass": offline["pass"],
            "live_pass": live["pass"] if live else False,
            "checks": {
                "offline": offline.get("checks"),
                "live": (live or {}).get("checks"),
            },
            "gold_consensus": offline.get("gold_consensus"),
        }
        (GATES / "divergence_tracer_proof.json").write_text(
            json.dumps(gate, indent=2), encoding="utf-8"
        )

    print(json.dumps(report, indent=2))
    return 0 if signoff else 1


if __name__ == "__main__":
    raise SystemExit(main())

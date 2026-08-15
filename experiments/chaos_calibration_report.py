#!/usr/bin/env python
"""Print A-G checker audit for persisted experiments (legacy vs calibrated)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.chaos.checker_audit import audit_invocation
from jlens_qwen.chaos.flags import evaluate_flags

OUT = ROOT / "experiments" / "out"


def flag_rate(tokens: list, prefix: str) -> float:
    if not tokens:
        return 0.0
    n = sum(1 for t in tokens if any(f.startswith(prefix) for f in (t.get("flags") or [])))
    return n / len(tokens)


def refire_tokens(tokens: list) -> list[str]:
    """Re-evaluate flags with calibrated evaluate_flags (simulation on stored payloads)."""
    union: set[str] = set()
    for t in tokens:
        payload = {k: v for k, v in t.items() if k not in ("flags", "step", "token_id")}
        if t.get("D") and not payload.get("D_boundary"):
            payload.pop("D", None)
        if payload.get("D_roll"):
            payload.pop("D_roll", None)
        if t.get("step") == 0 and payload.get("D"):
            payload["D_boundary"] = True
        union.update(evaluate_flags(payload))
    return sorted(union)


def main() -> int:
    names = [
        "aider_chaos_proof_clean_latest.json",
        "aider_chaos_proof_lint_stress_latest.json",
        "aider_live_panic_demo_scheduler_latest.json",
    ]
    for name in names:
        p = OUT / name
        if not p.is_file():
            print(f"SKIP {name}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        inv = data["invocations"][0]
        toks = (inv.get("chaos") or {}).get("tokens") or []
        print(f"\n=== {name} (inv#1, {len(toks)} tokens) ===")
        print(
            f"  LEGACY flags: C={flag_rate(toks,'C_'):.0%} "
            f"D={flag_rate(toks,'D_'):.0%} B={flag_rate(toks,'B_'):.0%} "
            f"A={flag_rate(toks,'A_'):.0%}"
        )
        sim = refire_tokens(toks)
        print(f"  CALIBRATED re-eval union: {sim}")
        audit = audit_invocation(inv.get("chaos"))
        for k, v in audit["checkers"].items():
            print(f"  {k:12} active={v['active']!s:5} fired={v['fired']!s:5} {v['severity']:4} | {v['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

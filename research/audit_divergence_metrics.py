#!/usr/bin/env python
"""Thermonuclear audit: divergence_hints vs offline scanners + token_id integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.chaos.flags import THRESH  # noqa: E402
from prefix_divergence_tracer import (  # noqa: E402
    _estimators,
    _first_both_flags,
    _first_mass_cross,
    _first_recipe_step,
    trace_invocation,
)


def _scan_hints_vs_tokens(chaos: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare live bus divergence_hints to offline re-scan of token rows."""
    hints = chaos.get("divergence_hints") or {}
    tokens = chaos.get("tokens") or []
    checks: list[dict[str, Any]] = []

    def _cmp(name: str, hint_key: str, scanned: int | None) -> None:
        hint_val = hints.get(hint_key)
        hint_i = int(hint_val) if hint_val is not None else None
        if hint_i is None and scanned is None:
            checks.append({"id": name, "pass": True, "detail": "both absent"})
            return
        if hint_i is None and scanned is not None:
            # Legacy chaos records finalized before divergence_hints shipped.
            checks.append(
                {
                    "id": name,
                    "pass": True,
                    "legacy": True,
                    "detail": f"hint absent (legacy) scanned={scanned}",
                }
            )
            return
        if hint_i is not None and scanned is None:
            checks.append(
                {
                    "id": name,
                    "pass": False,
                    "detail": f"hint={hint_i} scanned={scanned}",
                }
            )
            return
        drift = abs(hint_i - scanned)
        checks.append(
            {
                "id": name,
                "pass": drift == 0,
                "detail": f"hint={hint_i} scanned={scanned} drift={drift}",
            }
        )

    d_lex_scan = _first_mass_cross(tokens, both=True)
    d_flg_scan = _first_both_flags(tokens)
    d_dist_scan = _first_mass_cross(tokens, distress=True)
    d_stub_scan = _first_mass_cross(tokens, stub=True)

    _cmp("A1_d_lexicon", "d_lexicon", d_lex_scan)
    _cmp("A2_d_flags", "d_flags", d_flg_scan)
    _cmp("A3_d_distress_only", "d_distress_only", d_dist_scan)
    _cmp("A4_d_stub_only", "d_stub_only", d_stub_scan)

    n_full = int(chaos.get("n_tokens_full") or len(chaos.get("token_ids") or []) or 0)
    if n_full > 0 and hints:
        b_d_n = sum(1 for t in tokens if "B_distress" in (t.get("flags") or []))
        b_s_n = sum(1 for t in tokens if "B_stub" in (t.get("flags") or []))
        # Rate uses full stream length (bus semantics).
        exp_d = b_d_n / n_full
        exp_s = b_s_n / n_full
        got_d = float(hints.get("b_distress_rate", -1))
        got_s = float(hints.get("b_stub_rate", -1))
        checks.append(
            {
                "id": "A5_b_distress_rate",
                "pass": abs(got_d - exp_d) < 1e-9,
                "detail": f"hint={got_d:.6f} scanned={exp_d:.6f} n_full={n_full}",
            }
        )
        checks.append(
            {
                "id": "A6_b_stub_rate",
                "pass": abs(got_s - exp_s) < 1e-9,
                "detail": f"hint={got_s:.6f} scanned={exp_s:.6f} n_full={n_full}",
            }
        )
    return checks


def audit_invocation(inv: dict[str, Any], *, experiment: str = "", task_slug: str = "") -> dict[str, Any]:
    chaos = inv.get("chaos") or {}
    tokens = chaos.get("tokens") or []
    token_ids = chaos.get("token_ids") or []

    hint_checks = _scan_hints_vs_tokens(chaos)
    traced = trace_invocation(
        inv,
        experiment=experiment,
        task_slug=task_slug,
        use_tokenizer=False,
    ).to_dict()

    d_lex, d_flg, _, _ = _estimators(chaos, tokens)
    d_recipe = _first_recipe_step(chaos)

    consensus_checks: list[dict[str, Any]] = []
    if traced.get("eligible") and traced.get("d_consensus") is not None:
        cands = [x for x in (d_lex, d_flg) if x is not None]
        exp_consensus = min(cands) if cands else None
        consensus_checks.append(
            {
                "id": "B1_d_consensus",
                "pass": traced["d_consensus"] == exp_consensus,
                "detail": f"traced={traced['d_consensus']} expected={exp_consensus}",
            }
        )
        if token_ids and traced.get("d_consensus") is not None:
            d = int(traced["d_consensus"])
            prefix = traced.get("prefix_token_ids") or []
            consensus_checks.append(
                {
                    "id": "B2_prefix_token_ids",
                    "pass": prefix == list(token_ids[:d]),
                    "detail": f"len(prefix)={len(prefix)} d={d} token_ids_len={len(token_ids)}",
                }
            )

    if d_recipe is not None and traced.get("d_consensus") is not None:
        lag = d_recipe - int(traced["d_consensus"])
        consensus_checks.append(
            {
                "id": "B3_recipe_lag",
                "pass": 0 <= lag <= 12,
                "detail": f"lag={lag} recipe={d_recipe} d={traced['d_consensus']}",
            }
        )

    token_checks: list[dict[str, Any]] = []
    ct = inv.get("completion_tokens") or chaos.get("n_tokens_full")
    if token_ids and ct is not None:
        token_checks.append(
            {
                "id": "C1_token_ids_len",
                "pass": len(token_ids) == int(ct),
                "detail": f"len={len(token_ids)} completion_tokens={ct}",
            }
        )

    # Sparse token rows should still carry token_id for flagged steps.
    if token_ids and tokens:
        by_step = {int(t["step"]): int(t["token_id"]) for t in tokens if "token_id" in t}
        mismatches = [
            (s, by_step[s], token_ids[s])
            for s in sorted(by_step)
            if s < len(token_ids) and by_step[s] != token_ids[s]
        ]
        token_checks.append(
            {
                "id": "C2_sparse_token_id_match",
                "pass": len(mismatches) == 0,
                "detail": f"mismatches={mismatches[:5]}",
            }
        )

    all_checks = hint_checks + consensus_checks + token_checks
    ok = all(c["pass"] for c in all_checks) if all_checks else True
    return {
        "pass": ok,
        "invocation_id": inv.get("id") or inv.get("invocation_id"),
        "eligible": traced.get("eligible"),
        "d_consensus": traced.get("d_consensus"),
        "confidence": traced.get("confidence"),
        "checks": all_checks,
        "traced": traced,
        "thresh": {"B_distress": THRESH["B_distress"], "B_stub": THRESH["B_stub"]},
    }


def audit_experiment(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    experiment = str(data.get("experiment") or path.stem)
    task_slug = str(data.get("task_slug") or "")
    invocations = data.get("invocations") or []
    results = [audit_invocation(inv, experiment=experiment, task_slug=task_slug) for inv in invocations]
    n_pass = sum(1 for r in results if r["pass"])
    return {
        "experiment": experiment,
        "path": str(path),
        "n_invocations": len(results),
        "n_pass": n_pass,
        "all_pass": n_pass == len(results),
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit divergence metrics accuracy")
    ap.add_argument("paths", nargs="*", help="experiment JSON files")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "gates" / "divergence_metrics_audit.json",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths]
    if not paths:
        candidates = [
            ROOT / "experiments" / "out" / "divergence_corpus_stress_b_latest.json",
            ROOT / "experiments" / "out" / "divergence_live_proof_latest.json",
        ]
        paths = [p for p in candidates if p.is_file()]

    if not paths:
        print("No experiment files to audit", file=sys.stderr)
        return 1

    reports = [audit_experiment(p) for p in paths]
    out = {
        "all_pass": all(r["all_pass"] for r in reports),
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"all_pass": out["all_pass"], "n_files": len(reports), "out": str(args.out)}, indent=2))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

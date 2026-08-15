#!/usr/bin/env python
"""Prove F-lens false-positive root cause and validate post-fix logic.

Run: python research/verify_f_lens_calibration.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.chaos.checker_audit import audit_invocation
from jlens_qwen.chaos.flags import THRESH
from jlens_qwen.chaos.metrics_f import f_alarm_from_metrics, lens_disagreement
from jlens_qwen.chaos.recipes import f_disagree_from_payload

OUT = ROOT / "experiments" / "out"
REPORT = ROOT / "research" / "results" / "f_lens_calibration_proof.json"


def _collect_f(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in rows:
        for pdata in (t.get("probes") or {}).values():
            f = (pdata or {}).get("F") if isinstance(pdata, dict) else None
            if isinstance(f, dict):
                out.append(f)
    return out


def _audit_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    invs = data.get("invocations") or []
    audits = [audit_invocation((inv.get("chaos") or {})) for inv in invs]
    f_fired = sum(1 for a in audits if a.get("checkers", {}).get("F", {}).get("fired"))
    legacy_atoms = 0
    legacy_overlaps: list[float] = []
    for inv in invs:
        toks = (inv.get("chaos") or {}).get("tokens") or []
        for f in _collect_f(toks):
            legacy_overlaps.append(float(f.get("overlap", 1.0)))
            payload = {"probes": {"16": {"F": f}}}
            if f_disagree_from_payload(payload):
                legacy_atoms += 1
    return {
        "file": path.name,
        "n_invocations": len(invs),
        "f_checker_fired": f_fired,
        "legacy_f_atom_tokens": legacy_atoms,
        "legacy_overlap_min": min(legacy_overlaps) if legacy_overlaps else None,
        "legacy_overlap_p50": (
            sorted(legacy_overlaps)[len(legacy_overlaps) // 2] if legacy_overlaps else None
        ),
        "has_overlap_sample": any(
            f.get("overlap_sample") is not None
            for inv in invs
            for f in _collect_f((inv.get("chaos") or {}).get("tokens") or [])
        ),
        "verdicts": [inv.get("verdict") for inv in invs[:5]],
    }


def _synthetic_proof() -> dict:
    """Math proof: naive mid-layer compare always alarms; sampling agree does not."""
    jlens = [101, 202, 303, 404, 505]
    sampling = [101, 202, 303, 404, 505]
    naive_ll = [901, 902, 903, 904, 905]
    vs_sample = lens_disagreement(jlens, sampling)
    vs_naive = lens_disagreement(jlens, naive_ll)
    return {
        "identical_jlens_sampling": {
            "overlap": vs_sample["overlap"],
            "f_alarm": f_alarm_from_metrics({"overlap_sample": vs_sample["overlap"]}),
            "expect_alarm": False,
        },
        "jlens_vs_naive_midlayer": {
            "overlap": vs_naive["overlap"],
            "f_alarm": f_alarm_from_metrics({"overlap_sample": vs_naive["overlap"]}),
            "expect_alarm": True,
            "note": "Old bug compared to naive logit-lens — always low at fitted mid-layer",
        },
        "threshold": THRESH["F_low_overlap"],
    }


def main() -> int:
    targets = [
        OUT / "aider_chaos_proof_clean_latest.json",
        OUT / "aider_live_panic_demo_scheduler_latest.json",
        OUT / "stem25_t01_stem25_alm_duration_match_latest.json",
    ]
    persisted = [p for p in targets if p.is_file()]
    legacy_stats = [_audit_file(p) for p in persisted]

    proof = {
        "bug": (
            "F compared J-lens (transport+norm+unembed) to naive same-layer logit-lens. "
            "With fitted J at L10/L16/L22, mid-layer naive readout intentionally diverges "
            "(gate4 G4.4 expects >=50% top-1 mismatch). Overlap<=0.4 fired on virtually "
            "every token → F_LENS recipe spam (2000+ fires) even on clean fibonacci."
        ),
        "fix": (
            "Compare J-lens top-k to sampling-head top-k (actual decode logits). "
            "Store overlap_sample + overlap_naive diagnostic. CAST uses overlap_sample only."
        ),
        "synthetic": _synthetic_proof(),
        "legacy_persisted_runs": legacy_stats,
        "legacy_f_fired_rate": (
            sum(x["f_checker_fired"] for x in legacy_stats)
            / max(sum(x["n_invocations"] for x in legacy_stats), 1)
        ),
        "unit_tests": "pytest tests/unit/test_chaos_metrics.py -k f_alarm",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(proof, indent=2), encoding="utf-8")

    print(json.dumps(proof, indent=2))
    print(f"\nWrote {REPORT}")

    # Success criteria for this script (offline):
    syn = proof["synthetic"]
    ok = (
        syn["identical_jlens_sampling"]["f_alarm"] is False
        and syn["jlens_vs_naive_midlayer"]["f_alarm"] is True
        and proof["legacy_f_fired_rate"] >= 0.9
    )
    print(f"\nVERDICT: {'VERIFIED (bug reproduced + fix logic sound)' if ok else 'INCONCLUSIVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

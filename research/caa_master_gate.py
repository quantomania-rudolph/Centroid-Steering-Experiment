#!/usr/bin/env python
"""CAA master gate — pipeline_ready / capture_ready / production_ready tiers."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.steering_vectors import discover_natural_captures  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import (  # noqa: E402
    CAA_CORPUS_ROOT,
    CORPUS_GLOB,
)

GATES = ROOT / "artifacts" / "gates"
CAA_ROOT = ROOT / "artifacts" / "caa" / "UNI_CAA01"
# Canonical corpus + legacy UNI_CAA01 (scheduler smoke) both count toward pool.
NATURAL_DISCOVERY_ROOTS = [CAA_CORPUS_ROOT, CAA_ROOT]
STATUS = GATES / "caa_steering_status.json"

SUB_GATES = {
    "provenance": "caa_provenance_schema_gate.json",
    "kv_unit": "kv_equivalence_unit_gate.json",
    "kv_live": "kv_equivalence_live_gate.json",
    "calm_tail": "caa_calm_tail_purity_gate.json",
    "natural_admission": "caa_natural_pair_admission_gate.json",
    "natural_live": "caa_natural_pair_live_gate.json",
    "neutral": "caa_neutral_purity_gate.json",
    "aggregate": "caa_aggregate_natural_only_gate.json",
    "identity": "caa_vector_identity_gate.json",
    "cast_stream_live": "cast_stream_live_gate.json",
    "cast_inject_live": "cast_inject_live_gate.json",
}


def _load(name: str) -> dict | None:
    p = GATES / name
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _passed(name: str) -> bool:
    data = _load(name)
    return bool(data and data.get("pass"))


def main() -> int:
    GATES.mkdir(parents=True, exist_ok=True)
    natural = discover_natural_captures(NATURAL_DISCOVERY_ROOTS)
    n_natural = len(natural)
    natural_by_root = {
        str(root): len(discover_natural_captures(root)) for root in NATURAL_DISCOVERY_ROOTS
    }

    try:
        sys.path.insert(0, str(ROOT / "research"))
        from prefix_divergence_tracer import trace_all_glob

        n_corpus_eligible = sum(
            1
            for t in trace_all_glob(CORPUS_GLOB, use_tokenizer=False)
            if t.eligible and t.d_consensus is not None
        )
    except Exception:
        n_corpus_eligible = None

    pipeline_parts = [
        "caa_provenance_schema_gate.json",
        "kv_equivalence_unit_gate.json",
        "caa_calm_tail_purity_gate.json",
        "caa_natural_pair_admission_gate.json",
        "caa_aggregate_natural_only_gate.json",
        "caa_vector_identity_gate.json",
        "caa_neutral_purity_gate.json",
    ]
    pipeline_ready = all(_passed(n) for n in pipeline_parts)

    kv_live_ok = _passed("kv_equivalence_live_gate.json")
    # Capture tooling ready: KV live + admission fixture path, even if older purity gates stale.
    capture_tooling_ready = kv_live_ok and _passed("caa_natural_pair_admission_gate.json")
    capture_ready = pipeline_ready and capture_tooling_ready

    production_ready = (
        capture_ready
        and n_natural >= 3
        and _passed("caa_neutral_purity_gate.json")
        and _passed("caa_vector_identity_gate.json")
        and _passed("caa_aggregate_natural_only_gate.json")
        and _passed("caa_natural_pair_live_gate.json")
    )

    blockers: list[str] = []
    if not pipeline_ready:
        missing = [n for n in pipeline_parts if not _passed(n)]
        blockers.append(f"pipeline gates missing/fail: {missing}")
    if not kv_live_ok:
        blockers.append("kv_equivalence_live_gate not pass (set CAA_KV_LIVE=1 to run)")
    if n_natural == 0:
        blockers.append(
            "n_natural_admitted=0 — need live XPU capture into pairs/natural/ "
            "(dry-run pair validation does not count)"
        )
    if n_natural < 3:
        blockers.append(f"need ≥3 admitted natural pairs for production_ready (have {n_natural})")
    if not _passed("caa_tau_sigma_calibration_gate.json"):
        blockers.append(
            "tau/sigma calibration gate v2 not pass (run prove_tau_sigma_oos.py after corpus build)"
        )
    else:
        cal = _load("caa_tau_sigma_calibration_gate.json")
        if cal and int(cal.get("version") or 1) < 2:
            blockers.append("tau/sigma gate version < 2 — rebuild panic corpus + recalibrate")
    if not _passed("cast_stream_live_gate.json"):
        blockers.append("CAST shadow stream gate not pass (set CAA_CAST_LIVE=1 + prove_cast_stream_live)")
    elif not _passed("cast_inject_live_gate.json"):
        blockers.append("CAST inject gate not pass (informational until inject desired in prod)")

    # Master-S1 / Master-S2
    sab = [
        {
            "id": "Master-S1",
            "pass": production_ready is False or n_natural >= 3,
            "detail": "practice-only cannot be production_ready",
        },
        {
            "id": "Master-S2",
            "pass": not (n_natural == 0 and production_ready),
            "detail": "n_natural=0 ⇒ production_ready false",
        },
    ]

    status = {
        "gate": "caa_master",
        "version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pass": pipeline_ready,  # master "pass" = pipeline infrastructure ready
        "pipeline_ready": pipeline_ready,
        "capture_ready": capture_ready,
        "capture_tooling_ready": capture_tooling_ready,
        "production_ready": production_ready,
        "n_natural_admitted": n_natural,
        "n_natural_by_root": natural_by_root,
        "n_practice_excluded": 1,
        "n_corpus_eligible": n_corpus_eligible,
        "blockers": blockers,
        "gate_artifacts": {k: f"artifacts/gates/{v}" for k, v in SUB_GATES.items()},
        "sabotage": sab,
        "ready_for": [
            x
            for x, ok in [
                ("unit / fixture pipeline testing", pipeline_ready),
                ("live XPU natural capture (tooling)", capture_tooling_ready),
                ("KV live proof + full pipeline", capture_ready),
                ("production CAST vectors", production_ready),
            ]
            if ok
        ]
        or ["complete Phase 0–7 unit gates"],
        "notes": [
            "n_natural_admitted counts on-disk pool-eligible activations only (not dry-run)",
            "Discovery roots: UNI_CAA01_corpus + legacy UNI_CAA01",
            "Practice pair is plumbing / KV fixture only — never pooled",
            "production_ready requires ≥3 live natural pairs + green purity/identity/aggregate gates",
        ],
    }
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if pipeline_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

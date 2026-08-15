#!/usr/bin/env python
"""Indisputable proof harness: dry-run mining + architecture checks for LC / master."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from caa_mine_natural_pairs import run_live  # noqa: E402
from jlens_qwen.caa.steering_vectors import discover_natural_captures  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import CAA_CORPUS_ROOT, CALM_TAILS, NATURAL_PAIRS  # noqa: E402

PROOF = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof"
LEGACY = ROOT / "artifacts/caa/UNI_CAA01"


def main() -> int:
    from jlens_qwen.caa.calm_tail_registry import list_calm_tails

    before_corpus = discover_natural_captures(CAA_CORPUS_ROOT)
    ready_tails = list_calm_tails(CALM_TAILS)

    # 1) Full dry-run on corpus
    rc = run_live(dry_run=True, unicaa01_corpus=True)
    gate = json.loads((ROOT / "artifacts/gates/caa_natural_pair_live_gate.json").read_text(encoding="utf-8"))
    admitted = list(gate.get("admitted_slugs") or [])
    rejections = list(gate.get("rejections") or [])
    hard = [r for r in rejections if r.get("reason") not in {"missing_calm_tail"}]

    # 2) Architecture checks
    disk_natural_corpus = discover_natural_captures(CAA_CORPUS_ROOT)
    disk_natural_legacy = discover_natural_captures(LEGACY)
    disk_natural_both = discover_natural_captures([CAA_CORPUS_ROOT, LEGACY])

    claims = [
        {
            "id": "DRY_gate_pass",
            "pass": bool(gate.get("pass")) and rc == 0,
            "detail": f"rc={rc} pass={gate.get('pass')}",
        },
        {
            "id": "DRY_validated_eq_ready_tails",
            "pass": len(admitted) == len(ready_tails),
            "detail": f"validated={len(admitted)} ready_tails={len(ready_tails)}",
        },
        {
            "id": "DRY_no_hard_rejects",
            "pass": len(hard) == 0,
            "detail": f"hard={hard[:5]}",
        },
        {
            "id": "DRY_missing_calm_only_quarantine",
            "pass": all(r.get("reason") == "missing_calm_tail" for r in rejections) if rejections else True,
            "detail": f"n_missing_calm={len(rejections)}",
        },
        {
            "id": "ARCH_natural_root_corpus",
            "pass": Path(gate.get("natural_root") or "") == NATURAL_PAIRS,
            "detail": str(gate.get("natural_root")),
        },
        {
            "id": "ARCH_dry_run_does_not_write_activations",
            "pass": len(disk_natural_corpus) == len(before_corpus),
            "detail": f"before={len(before_corpus)} after={len(disk_natural_corpus)}",
        },
        {
            "id": "METRIC_mode_is_dry_run",
            "pass": (gate.get("environment") or {}).get("mode") == "dry_run",
            "detail": str((gate.get("environment") or {}).get("mode")),
        },
    ]

    # 3) Master gate snapshot after discovery fix
    import caa_master_gate

    master_rc = caa_master_gate.main()
    master = json.loads((ROOT / "artifacts/gates/caa_steering_status.json").read_text(encoding="utf-8"))
    claims.append(
        {
            "id": "ARCH_master_discovers_both_roots",
            "pass": isinstance(master.get("n_natural_by_root"), dict)
            and len(master.get("n_natural_by_root") or {}) >= 2,
            "detail": (
                f"corpus={len(disk_natural_corpus)} legacy={len(disk_natural_legacy)} "
                f"both={len(disk_natural_both)} by_root={master.get('n_natural_by_root')}"
            ),
        }
    )
    claims.append(
        {
            "id": "MASTER_counts_on_disk_only",
            "pass": int(master.get("n_natural_admitted") or 0) == len(disk_natural_both),
            "detail": (
                f"master_n={master.get('n_natural_admitted')} "
                f"discover_both={len(disk_natural_both)} "
                f"by_root={master.get('n_natural_by_root')}"
            ),
        }
    )
    claims.append(
        {
            "id": "MASTER_explains_not_production",
            "pass": master.get("production_ready") is False,
            "detail": (
                f"production_ready={master.get('production_ready')} "
                f"capture_tooling_ready={master.get('capture_tooling_ready')} "
                f"blockers={master.get('blockers')[:3]}"
            ),
        }
    )
    claims.append(
        {
            "id": "MASTER_capture_tooling_ready",
            "pass": bool(master.get("capture_tooling_ready")),
            "detail": f"capture_tooling_ready={master.get('capture_tooling_ready')}",
        }
    )

    payload = {
        "pass": all(c["pass"] for c in claims),
        "dry_run": {
            "n_admitted": gate.get("n_admitted"),
            "n_rejected": gate.get("n_rejected"),
            "n_eligible": gate.get("n_eligible"),
            "admitted_slugs": admitted,
            "rejections": rejections,
        },
        "master": {
            "pipeline_ready": master.get("pipeline_ready"),
            "capture_ready": master.get("capture_ready"),
            "capture_tooling_ready": master.get("capture_tooling_ready"),
            "production_ready": master.get("production_ready"),
            "n_natural_admitted": master.get("n_natural_admitted"),
            "n_natural_by_root": master.get("n_natural_by_root"),
            "blockers": master.get("blockers"),
            "master_rc": master_rc,
        },
        "claims": claims,
        "definitions": {
            "dry_run": (
                "Loads tokenizer, rebuilds calm tails from curated files, runs validate_pair "
                "for every eligible panic+calm candidate. Does NOT load the model or write activations."
            ),
            "live_capture": (
                "Same validation, then loads Qwen on XPU, reconstructs prompt token ids, "
                "runs shared-KV activation capture for calm vs bad completions, writes "
                "pairs/natural/<slug>/{activations_at_d.pt, manifests, admission_gate}."
            ),
            "production_ready": (
                "Master tier: capture_ready + ≥3 on-disk pool-eligible natural pairs + "
                "green neutral/identity/aggregate/live gates. Dry-run alone never flips this."
            ),
        },
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    out = PROOF / "G27_miner_architecture_proof.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"pass": payload["pass"], "claims": claims, "out": str(out)}, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

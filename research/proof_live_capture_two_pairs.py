#!/usr/bin/env python
"""Live-capture proof: mine exactly N corpus pairs on XPU and verify on-disk admission."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from caa_mine_natural_pairs import run_live  # noqa: E402
from jlens_qwen.caa.steering_vectors import discover_natural_captures  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import CAA_CORPUS_ROOT, NATURAL_PAIRS  # noqa: E402

PROOF = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof"


def main() -> int:
    n = int(os.environ.get("CAA_LIVE_PROOF_N", "2"))
    before = discover_natural_captures(CAA_CORPUS_ROOT)
    os.environ["CAA_MINE_LIVE"] = "1"
    rc = run_live(dry_run=False, unicaa01_corpus=True, device=os.environ.get("CAA_DEVICE", "xpu"), max_pairs=n)
    gate = json.loads((ROOT / "artifacts/gates/caa_natural_pair_live_gate.json").read_text(encoding="utf-8"))
    after = discover_natural_captures(CAA_CORPUS_ROOT)

    # Inspect written dirs
    dirs = sorted(p for p in NATURAL_PAIRS.iterdir() if p.is_dir()) if NATURAL_PAIRS.is_dir() else []
    dir_reports = []
    for d in dirs:
        act = d / "activations_at_d.pt"
        adm = d / "admission_gate.json"
        man = d / "capture_manifest.json"
        adm_data = json.loads(adm.read_text(encoding="utf-8")) if adm.is_file() else {}
        dir_reports.append(
            {
                "slug": d.name,
                "has_activations": act.is_file(),
                "has_manifest": man.is_file(),
                "admission_pass": bool(adm_data.get("pass")),
                "pool_eligible": bool(adm_data.get("pool_eligible")),
                "source": adm_data.get("source"),
            }
        )

    claims = [
        {
            "id": "LIVE_rc_ok",
            "pass": rc == 0 and bool(gate.get("pass")),
            "detail": f"rc={rc} gate_pass={gate.get('pass')} mode={(gate.get('environment') or {}).get('mode')}",
        },
        {
            "id": "LIVE_mode_capture",
            "pass": (gate.get("environment") or {}).get("mode") == "capture",
            "detail": str((gate.get("environment") or {}).get("mode")),
        },
        {
            "id": "LIVE_admitted_ge_n",
            "pass": int(gate.get("n_admitted") or 0) >= n,
            "detail": f"n_admitted={gate.get('n_admitted')} need>={n} slugs={gate.get('admitted_slugs')}",
        },
        {
            "id": "LIVE_discover_increased",
            "pass": len(after) >= len(before) + n,
            "detail": f"before={len(before)} after={len(after)} need+{n}",
        },
        {
            "id": "LIVE_dirs_pool_eligible",
            "pass": sum(1 for r in dir_reports if r["pool_eligible"] and r["has_activations"]) >= n,
            "detail": dir_reports,
        },
    ]

    # Master should now see corpus captures
    import caa_master_gate

    caa_master_gate.main()
    master = json.loads((ROOT / "artifacts/gates/caa_steering_status.json").read_text(encoding="utf-8"))
    claims.append(
        {
            "id": "MASTER_sees_corpus_captures",
            "pass": int((master.get("n_natural_by_root") or {}).get(str(CAA_CORPUS_ROOT), 0) or 0) >= n
            or int(master.get("n_natural_admitted") or 0) >= n,
            "detail": {
                "n_natural_admitted": master.get("n_natural_admitted"),
                "n_natural_by_root": master.get("n_natural_by_root"),
                "production_ready": master.get("production_ready"),
                "capture_tooling_ready": master.get("capture_tooling_ready"),
            },
        }
    )

    payload = {
        "pass": all(c["pass"] for c in claims),
        "n_requested": n,
        "gate_n_admitted": gate.get("n_admitted"),
        "admitted_slugs": gate.get("admitted_slugs"),
        "rejections": [r for r in (gate.get("rejections") or []) if r.get("reason") != "missing_calm_tail"][:20],
        "dir_reports": dir_reports,
        "master": {
            "n_natural_admitted": master.get("n_natural_admitted"),
            "n_natural_by_root": master.get("n_natural_by_root"),
            "production_ready": master.get("production_ready"),
            "capture_tooling_ready": master.get("capture_tooling_ready"),
            "blockers": master.get("blockers"),
        },
        "claims": claims,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    out = PROOF / "G28_live_capture_proof.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"pass": payload["pass"], "claims": claims, "out": str(out)}, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

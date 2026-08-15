"""Baseline: show v4 admits fail new v5 form/prose gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from validate_aider_listing import validate_aider_listing  # noqa: E402
from validate_calm_curator_output import brief_from_dict  # noqa: E402
from validate_narrative_purity import validate_narrative_purity  # noqa: E402

CALM = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "calm_tails"
BRIEFS = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_briefs"
PROOF = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_hardening_proof"


def main() -> int:
    rows = []
    for txt in sorted(CALM.glob("unicaa01_p*.txt")):
        meta = json.loads(txt.with_suffix(".meta.json").read_text(encoding="utf-8"))
        brief_id = str(meta["brief_id"])
        brief = brief_from_dict(json.loads((BRIEFS / f"{brief_id}.json").read_text(encoding="utf-8")))
        text = txt.read_text(encoding="utf-8")
        narr = {
            r.gate_id: r.pass_
            for r in validate_narrative_purity(
                full_text=text,
                prefix_text=brief.prefix_lock.prefix_text,
                pair_id=brief.pair_id,
            )
        }
        fmt = {
            r.gate_id: r.pass_
            for r in validate_aider_listing(
                full_text=text,
                task_dir=brief.output_contract.task_dir,
            )
        }
        failed = [k for k, v in {**narr, **fmt}.items() if not v]
        rows.append(
            {
                "brief_id": brief_id,
                "pair_id": brief.pair_id,
                "pipeline_version": meta.get("pipeline_version"),
                "failed_new_gates": failed,
                "narr": narr,
                "fmt": fmt,
            }
        )
    payload = {
        "phase": "pre_redo_baseline",
        "expect": "all four fail at least one new v5 form/prose gate",
        "n": len(rows),
        "all_fail_new_gates": all(bool(r["failed_new_gates"]) for r in rows),
        "rows": rows,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    (PROOF / "G15_form_gates_baseline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_fail_new_gates"] and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""G15 after-proof: v5 form/prose gates pass on live admits + unit regression."""
from __future__ import annotations

import json
import subprocess
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

NEW_GATES = (
    "NARR_continuation_grammar",
    "NARR_balanced_backticks",
    "FMT_aider_path_outside_fence",
    "FMT_no_path_comment_in_fence",
)


def main() -> int:
    baseline_path = PROOF / "G15_form_gates_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {}

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
            for r in validate_aider_listing(full_text=text, task_dir=brief.output_contract.task_dir)
        }
        merged = {**narr, **fmt}
        rows.append(
            {
                "brief_id": brief_id,
                "pair_id": brief.pair_id,
                "pipeline_version": meta.get("pipeline_version"),
                "new_gates": {k: merged.get(k) for k in NEW_GATES},
                "all_new_gates_pass": all(merged.get(k) for k in NEW_GATES),
                "path_outside_fence_line": next(
                    (
                        ln.strip()
                        for ln in text.splitlines()
                        if ln.strip().endswith("/main.py")
                    ),
                    "",
                ),
                "narrative_first_line": text.splitlines()[0] if text.splitlines() else "",
            }
        )

    unit = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_form_gates_v5.py",
            "tests/unit/test_narrative_purity.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    verify = subprocess.run(
        [sys.executable, "research/verify_admitted_tail.py", "--all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        verify_payload = json.loads(verify.stdout)
    except json.JSONDecodeError:
        verify_payload = {"raw": verify.stdout[-2000:], "returncode": verify.returncode}

    payload = {
        "phase": "post_redo_proof",
        "pipeline_version_required": 5,
        "baseline_all_failed_new_gates": baseline.get("all_fail_new_gates"),
        "baseline_rows": baseline.get("rows"),
        "after_all_new_gates_pass": all(r["all_new_gates_pass"] for r in rows) and bool(rows),
        "after_all_pipeline_v5": all(int(r["pipeline_version"] or 0) >= 5 for r in rows) and bool(rows),
        "rows": rows,
        "unit_tests": {
            "returncode": unit.returncode,
            "stdout_tail": unit.stdout[-500:],
        },
        "verify_admitted": verify_payload,
        "pass": (
            bool(baseline.get("all_fail_new_gates"))
            and all(r["all_new_gates_pass"] for r in rows)
            and all(int(r["pipeline_version"] or 0) >= 5 for r in rows)
            and unit.returncode == 0
            and int(verify_payload.get("n_pass") or 0) == int(verify_payload.get("n_files") or -1)
            and int(verify_payload.get("n_files") or 0) > 0
        ),
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    out = PROOF / "G15_form_gates_after.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("pass", "after_all_new_gates_pass", "after_all_pipeline_v5", "unit_tests", "verify_admitted") if k in payload} | {
        "n_rows": len(rows),
        "proof_path": str(out),
        "baseline_all_failed_new_gates": payload["baseline_all_failed_new_gates"],
    }, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

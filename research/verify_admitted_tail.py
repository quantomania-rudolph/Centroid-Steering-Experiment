#!/usr/bin/env python
"""Re-run L0-L6 gates on admitted calm tail files on disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py  # noqa: E402
from calm_curator_demo_check import (  # noqa: E402
    check_invariants,
    check_no_hardcode,
    run_demo_check,
    run_demo_stress,
    run_quality_suite,
)
from validate_aider_listing import validate_aider_listing  # noqa: E402
from validate_calm_curator_output import brief_from_dict, validate_calm_curator_output  # noqa: E402
from validate_narrative_purity import validate_narrative_purity  # noqa: E402

CALM_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "calm_tails"
BRIEFS_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_briefs"
PROOF = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_hardening_proof"


def verify_one(txt_path: Path) -> dict[str, object]:
    meta_path = txt_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    brief_id = str(meta.get("brief_id") or "")
    brief_path = BRIEFS_DIR / f"{brief_id}.json"
    if not brief_path.is_file():
        return {"file": str(txt_path), "pass": False, "error": f"no brief {brief_id}"}

    brief = brief_from_dict(json.loads(brief_path.read_text(encoding="utf-8")))
    text = txt_path.read_text(encoding="utf-8")
    main_py = extract_main_py(text)

    l0 = validate_calm_curator_output(
        brief=brief,
        candidate_text=text[len(brief.prefix_lock.prefix_text) :] if text.startswith(brief.prefix_lock.prefix_text) else text,
        attempt_repair=False,
        sanitize=False,
        run_miner_parity=False,
        run_demo_oracle=False,
    )
    checks = {c.id: c.pass_ for c in l0.checks}
    demo = run_demo_check(main_py=main_py, pair_id=brief.pair_id)
    checks["RUN_demo_oracle"] = demo.pass_
    stress = run_demo_stress(main_py=main_py, pair_id=brief.pair_id)
    checks["RUN_demo_stress"] = stress.pass_
    checks["SEM_invariants"] = check_invariants(main_py=main_py, pair_id=brief.pair_id).pass_
    checks["SEM_no_hardcode"] = check_no_hardcode(main_py=main_py, pair_id=brief.pair_id).pass_
    for nr in validate_narrative_purity(full_text=text, prefix_text=brief.prefix_lock.prefix_text, pair_id=brief.pair_id):
        checks[nr.gate_id] = nr.pass_
    for lr in validate_aider_listing(full_text=text, task_dir=brief.output_contract.task_dir):
        checks[lr.gate_id] = lr.pass_
    checks["RUN_quality_suite"] = run_quality_suite(
        main_py=main_py or "", pair_id=brief.pair_id, require_configured=True
    ).pass_
    l6 = validate_calm_curator_output(
        brief=brief,
        candidate_text=text,
        attempt_repair=False,
        sanitize=False,
        parity_only=True,
        run_miner_parity=True,
        run_demo_oracle=False,
    )
    checks["PAIR_validate"] = l6.pass_

    meta_ok = (
        int(meta.get("pipeline_version") or 0) >= 5
        and meta.get("status") == "admitted"
        and not meta.get("repaired")
        and not meta.get("bridge_used")
        and not meta.get("sanitized")
        and not meta.get("qa_skipped")
        and not (meta.get("qa") or {}).get("llm_skipped")
        and not (meta.get("qa") or {}).get("qa_skipped")
        and bool((meta.get("judge") or (meta.get("qa") or {}).get("judge") or {}).get("ran"))
    )
    gate_ok = all(checks.get(k) for k in checks)
    return {
        "file": str(txt_path),
        "brief_id": brief_id,
        "pass": gate_ok and meta_ok,
        "checks": checks,
        "meta_ok": meta_ok,
        "pipeline_version": meta.get("pipeline_version"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--path", type=Path)
    ap.add_argument(
        "--admitted-only",
        action="store_true",
        help="Skip quarantined/rejected metas; verify only downstream-ready admits",
    )
    args = ap.parse_args()

    paths = [args.path] if args.path else sorted(CALM_DIR.glob("unicaa01_*.txt"))
    if args.admitted_only:
        filtered: list[Path] = []
        for p in paths:
            if not p.is_file():
                continue
            meta_path = p.with_suffix(".meta.json")
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") == "admitted" and int(meta.get("pipeline_version") or 0) >= 5:
                filtered.append(p)
        paths = filtered
    results = [verify_one(p) for p in paths if p.is_file()]
    payload = {
        "n_files": len(results),
        "n_pass": sum(1 for r in results if r.get("pass")),
        "admitted_only": bool(args.admitted_only),
        "results": results,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    out_name = "G10_verify_admitted.json" if args.admitted_only or args.all else "G10_verify_admitted.json"
    if args.admitted_only:
        out_name = "G22_verify_downstream_ready.json"
    (PROOF / out_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.admitted_only:
        (PROOF / "G10_verify_admitted.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["n_pass"] == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())

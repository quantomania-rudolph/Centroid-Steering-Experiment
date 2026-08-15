"""L5 semantic QA — delegates to v4 quality loop (judge required, no auto-approve)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py  # noqa: E402
from calm_curator_demo_check import (  # noqa: E402
    check_invariants,
    check_no_hardcode,
    run_demo_check,
    run_quality_suite,
)
from calm_curator_qa_review import mechanical_qa_preflight  # noqa: E402
from calm_curator_schemas import CuratorBrief  # noqa: E402
from openrouter_curator_client import OpenRouterCuratorClient  # noqa: E402
from validate_narrative_purity import narrative_all_pass, validate_narrative_purity  # noqa: E402


def run_invariant_proof(*, brief: CuratorBrief, full_text: str) -> dict[str, Any]:
    """Deterministic L1–L4 proof — LLM cannot override failures."""
    main_py = extract_main_py(full_text)
    results: dict[str, Any] = {"pass": False, "checks": {}, "fix_hints": []}

    demo = run_demo_check(main_py=main_py, pair_id=brief.pair_id) if main_py else None
    inv = check_invariants(main_py=main_py, pair_id=brief.pair_id) if main_py else None
    hc = check_no_hardcode(main_py=main_py, pair_id=brief.pair_id) if main_py else None
    narr = validate_narrative_purity(
        full_text=full_text,
        prefix_text=brief.prefix_lock.prefix_text,
        pair_id=brief.pair_id,
    )

    results["checks"]["RUN_demo_oracle"] = bool(demo and demo.pass_)
    results["checks"]["SEM_invariants"] = bool(inv and inv.pass_)
    results["checks"]["SEM_no_hardcode"] = bool(hc and hc.pass_)
    results["checks"]["NARR_all"] = narrative_all_pass(narr)
    suite = run_quality_suite(main_py=main_py or "", pair_id=brief.pair_id, require_configured=True)
    results["checks"]["RUN_quality_suite"] = suite.pass_
    for nr in narr:
        results["checks"][nr.gate_id] = nr.pass_

    if not main_py:
        results["fix_hints"].append("missing fenced main.py")
    if demo and not demo.pass_:
        results["fix_hints"].append(f"demo oracle: {demo.error or demo.actual_repr}")
    if inv and not inv.pass_:
        results["fix_hints"].append(f"invariants: {inv.detail}")
    if hc and not hc.pass_:
        results["fix_hints"].append(f"hardcode: {hc.detail}")
    for nr in narr:
        if not nr.pass_:
            results["fix_hints"].append(f"{nr.gate_id}: {nr.detail}")
    if not suite.pass_:
        results["fix_hints"].append(f"RUN_quality_suite: {suite.detail}")

    results["pass"] = all(
        [
            results["checks"]["RUN_demo_oracle"],
            results["checks"]["SEM_invariants"],
            results["checks"]["SEM_no_hardcode"],
            results["checks"]["NARR_all"],
            results["checks"]["RUN_quality_suite"],
        ]
    )
    if results["pass"]:
        results["scores"] = {
            "correctness": 3,
            "no_stubs": 3,
            "task_fit": 3,
            "completeness": 3,
            "narrative_quality": 3,
        }
    return results


def run_semantic_qa(
    *,
    brief: CuratorBrief,
    full_text: str,
    gate_report: dict[str, Any],
    demo_check: dict[str, Any],
    client: OpenRouterCuratorClient | None = None,
    brief_id: str | None = None,
    skip_llm_if_proof_passes: bool = False,
    suite_passed: bool = True,
    suite_detail: str = "",
) -> dict[str, Any]:
    """L5 entrypoint — always runs quality loop (judge required)."""
    _ = skip_llm_if_proof_passes  # v4: auto-approve removed; kept for call-site compat
    # Circular-dependency: quality_loop imports run_invariant_proof from this module.
    from calm_curator_quality_loop import run_quality_loop  # noqa: PLC0415

    loop = run_quality_loop(
        brief=brief,
        full_text=full_text,
        gate_report=gate_report,
        demo_check=demo_check,
        client=client,
        brief_id=brief_id,
        suite_passed=suite_passed,
        suite_detail=suite_detail,
    )
    verdict = str(loop.get("verdict") or "reject")
    if verdict == "needs_edit":
        verdict = "revise"
    judge = loop.get("judge") or {}
    return {
        "verdict": verdict,
        "scores": loop.get("scores") or judge.get("scores") or {},
        "reasons": list(loop.get("reasons") or judge.get("reasons") or []),
        "fix_hints": list(loop.get("fix_hints") or judge.get("edit_brief") or []),
        "semantic_qa": True,
        "proof": loop.get("proof"),
        "judge": judge,
        "judge.ran": bool(judge.get("ran")),
        "llm_skipped": False,
        "edit_count": loop.get("edit_count", 0),
        "judge_calls": loop.get("judge_calls", 0),
        "quality_loop": loop,
    }

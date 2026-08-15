"""DeepSeek QA review for calm curator candidates (minimal QaBrief)."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py, tail_after_prefix  # noqa: E402
from calm_curator_schemas import CuratorBrief, QaBrief, Verdict  # noqa: E402
from openrouter_budget import BudgetExhausted
from openrouter_curator_client import OpenRouterCuratorClient, qa_model  # noqa: E402
from validate_aider_listing import validate_aider_listing  # noqa: E402
from validate_calm_curator_output import brief_from_dict  # noqa: E402
from validate_narrative_purity import validate_narrative_purity  # noqa: E402

QA_SYSTEM = """You are a strict calm-tail QA reviewer for prefix-locked CAA pairs.
Reply with JSON only: {"verdict":"approve"|"revise"|"reject","reasons":["..."],"fix_hints":["..."]}

Conservative policy — avoid false positives (wrong approve):
- Default to reject or revise when uncertain.
- Approve ONLY if the extracted main.py plausibly fixes the task and success criteria.
- Reject if: stub/TODO/NotImplementedError in code, prefix would change, listing missing, or fix is cosmetic only.
- Mechanical gates already passed; your job is semantic correctness of the fix, not prefix math."""


def mechanical_qa_preflight(
    *,
    brief: CuratorBrief,
    full_text: str,
) -> tuple[bool, list[str]]:
    """Deterministic checks QA must not override with approve."""
    issues: list[str] = []
    main_py = extract_main_py(full_text)
    if not main_py:
        issues.append("missing fenced main.py block")
    else:
        try:
            ast.parse(main_py)
        except SyntaxError as exc:
            issues.append(f"main.py syntax error: {exc}")
        for pat in ("NotImplementedError", "raise NotImplemented", "TODO", "FIXME"):
            if pat.lower() in main_py.lower():
                issues.append(f"forbidden token in code: {pat}")

    task_dir = brief.output_contract.task_dir
    for lr in validate_aider_listing(full_text=full_text, task_dir=task_dir):
        if not lr.pass_:
            issues.append(f"{lr.gate_id}: {lr.detail}")

    for nr in validate_narrative_purity(
        full_text=full_text,
        prefix_text=brief.prefix_lock.prefix_text,
        pair_id=brief.pair_id,
    ):
        if nr.gate_id in {"NARR_continuation_grammar", "NARR_balanced_backticks"} and not nr.pass_:
            issues.append(f"{nr.gate_id}: {nr.detail}")

    if not full_text.startswith(brief.prefix_lock.prefix_text):
        issues.append("completion does not start with locked prefix text")

    return len(issues) == 0, issues


def build_qa_brief(
    *,
    brief: CuratorBrief,
    full_text: str,
    gate_report: dict[str, Any],
    demo_check: dict[str, Any] | None = None,
) -> QaBrief:
    main_py = extract_main_py(full_text)
    return QaBrief(
        pair_id=brief.pair_id,
        d=brief.prefix_lock.d,
        task_summary=brief.task_summary,
        success_criteria=brief.success_criteria,
        prefix_text=brief.prefix_lock.prefix_text,
        bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
        forbidden_patterns=brief.divergence_context.forbidden_patterns,
        extracted_main_py=main_py[:1500],
        tail_after_prefix=tail_after_prefix(full_text, brief.prefix_lock.prefix_text)[:800],
        mechanical_gate_summary=json.dumps(
            {
                "pass": gate_report.get("pass"),
                "bridge_used": gate_report.get("bridge_used"),
                "sanitized": gate_report.get("sanitized"),
                "repaired": gate_report.get("repaired"),
                "failed": [c for c in gate_report.get("checks", []) if not c.get("pass")],
            },
            indent=0,
        )[:600],
        gate_failures_if_revise="",
        demo_check=demo_check,
    )


def _parse_verdict(content: str) -> tuple[Verdict, list[str], list[str]]:
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return "reject", ["unparseable QA response"], []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "reject", ["invalid JSON from QA"], []
    verdict = str(data.get("verdict") or "reject").lower()
    if verdict not in ("approve", "revise", "reject"):
        verdict = "reject"
    reasons = [str(x) for x in data.get("reasons") or []]
    hints = [str(x) for x in data.get("fix_hints") or []]
    return verdict, reasons, hints  # type: ignore[return-value]


def run_qa_review(
    *,
    brief: CuratorBrief | dict[str, Any],
    full_text: str,
    gate_report: dict[str, Any],
    demo_check: dict[str, Any] | None = None,
    client: OpenRouterCuratorClient | None = None,
    brief_id: str | None = None,
) -> dict[str, Any]:
    if isinstance(brief, dict):
        brief = brief_from_dict(brief)

    if demo_check is not None and not demo_check.get("pass"):
        return {
            "verdict": "reject",
            "reasons": ["demo_check failed before QA"],
            "fix_hints": [f"demo() must return {demo_check.get('expected_repr', '')}"],
            "raw": "",
            "model": "mechanical",
            "preflight_only": True,
        }

    pre_ok, pre_issues = mechanical_qa_preflight(brief=brief, full_text=full_text)
    if not pre_ok:
        return {
            "verdict": "reject",
            "reasons": ["mechanical_qa_preflight failed"] + pre_issues,
            "fix_hints": [],
            "raw": "",
            "model": "mechanical",
            "preflight_only": True,
        }

    cli = client or OpenRouterCuratorClient()
    qb = build_qa_brief(
        brief=brief, full_text=full_text, gate_report=gate_report, demo_check=demo_check
    )
    user = json.dumps(qb.to_dict(), indent=2)
    bid = brief_id or brief.brief_id
    try:
        cli.ledger.record_qa(bid)
    except BudgetExhausted:
        raise
    resp = cli.complete_with_retry(
        model=qa_model(cli.config).id,
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=256,
        role="qa",
        brief_id=brief_id or brief.brief_id,
    )
    if not resp.ok:
        return {
            "verdict": "reject",
            "reasons": [f"QA call failed: {resp.error_class} {resp.detail}"],
            "fix_hints": [],
            "raw": resp.detail,
            "model": resp.model,
        }
    verdict, reasons, hints = _parse_verdict(resp.content)

    if verdict == "approve":
        post_ok, post_issues = mechanical_qa_preflight(brief=brief, full_text=full_text)
        if not post_ok:
            return {
                "verdict": "reject",
                "reasons": ["QA approve overridden by postflight"] + post_issues,
                "fix_hints": hints,
                "raw": resp.content,
                "model": resp.model,
                "qa_false_positive_blocked": True,
            }

    return {
        "verdict": verdict,
        "reasons": reasons,
        "fix_hints": hints,
        "raw": resp.content,
        "model": resp.model,
    }

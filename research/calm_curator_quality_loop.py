"""L5 quality loop — deterministic proof + required judge + optional edit rounds."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py, tail_after_prefix  # noqa: E402
from calm_curator_demo_check import run_quality_suite  # noqa: E402
from calm_curator_qa_review import mechanical_qa_preflight  # noqa: E402
from calm_curator_schemas import CuratorBrief  # noqa: E402
from calm_curator_semantic_qa import run_invariant_proof  # noqa: E402
from openrouter_budget import BudgetExhausted  # noqa: E402
from openrouter_curator_client import (  # noqa: E402
    OpenRouterCuratorClient,
    generation_models,
    qa_model,
)
from unicaa01_oracle_invariants import curator_prompt_constraints  # noqa: E402

QualityVerdict = Literal["approve", "needs_edit", "reject", "revise"]

JUDGE_SYSTEM = """Judge calm-tail code. Suite+proof already passed.
Score ONLY task_summary, success_criteria, invariants, and suite scope.
Do NOT demand extra features, frameworks, tools, or full-RFC parsers.
Approve suite-passing short stdlib code that handles the suite edges (score≥4).
Thin hardcoded demo hacks → needs_edit.
Form is in scope: narrative_quality must drop below 4 for broken prose, unclosed backticks,
or prompt-paste imperatives (Implement/Write/…). Path must be outside the fence (not # path inside).
JSON only:
{"verdict":"approve"|"needs_edit"|"reject","scores":{"robustness":1-5,"clarity":1-5,"edge_handling":1-5,"task_fidelity":1-5,"narrative_quality":1-5},"reasons":["short"],"edit_brief":["scoped fix"]}
Max 3 reasons, max 3 edit_brief items. Keep edits Qwen-achievable (short stdlib)."""

EDIT_SYSTEM = """Edit calm-tail suffix only (after locked prefix).
Strict aider format: prose line, then path OUTSIDE fence (`task/main.py`), then ```python file```.
Close open backticks from the prefix. No Implement/Write/Create prompt-paste in prose.
Stdlib only. Short Qwen-achievable fix. Return suffix only — do not repeat the prefix."""


def _quality_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("quality") or {}


def _parse_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def _min_score(cfg: dict[str, Any]) -> int:
    return int(_quality_cfg(cfg).get("min_score") or 4)


def _score_keys() -> tuple[str, ...]:
    return ("robustness", "clarity", "edge_handling", "task_fidelity", "narrative_quality")


def _clamp_score(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, n))


def _apply_judge_rules(scores: dict[str, Any], min_score: int) -> QualityVerdict:
    for key in _score_keys():
        if int(scores.get(key) or 0) < min_score:
            return "needs_edit"
    if all(int(scores.get(k) or 0) >= min_score for k in _score_keys()):
        return "approve"
    return "needs_edit"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def _is_empty_content(content: str) -> bool:
    c = (content or "").strip()
    return c in {"", "None", "null", "none", "{}"}


def _normalize_str_list(value: Any, *, limit: int = 3, max_len: int = 160) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:limit]:
        s = str(item).strip()
        if s:
            out.append(s[:max_len])
    return out


def _judge_payload(
    *,
    brief: CuratorBrief,
    full_text: str,
    proof: dict[str, Any],
    suite_detail: str,
    demo_check: dict[str, Any],
) -> dict[str, Any]:
    main_py = extract_main_py(full_text) or ""
    tail = tail_after_prefix(full_text, brief.prefix_lock.prefix_text)
    return {
        "pair_id": brief.pair_id,
        "task": brief.task_summary[:200],
        "success": brief.success_criteria[:160],
        "constraints": curator_prompt_constraints(brief.pair_id)[:400],
        "suite": suite_detail[:200],
        "suite_ok": True,
        "proof_ok": bool(proof.get("pass")),
        "expect": brief.demo_oracle.expect_repr if brief.demo_oracle else "",
        "demo_ok": bool(demo_check.get("pass")),
        "main_py": main_py[:1200],
        "narrative": tail[:240],
        "scope": "Approve if suite edges are handled with short stdlib code. No extra features.",
    }


def _call_judge(
    *,
    cli: OpenRouterCuratorClient,
    brief: CuratorBrief,
    full_text: str,
    gate_report: dict[str, Any],
    demo_check: dict[str, Any],
    proof: dict[str, Any],
    suite_detail: str,
    brief_id: str,
) -> dict[str, Any]:
    _ = gate_report  # mechanical gates already enforced upstream
    cfg = cli.config
    qcfg = cfg.get("qa") or {}
    model = qa_model(cfg)
    payload = _judge_payload(
        brief=brief,
        full_text=full_text,
        proof=proof,
        suite_detail=suite_detail,
        demo_check=demo_check,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]
    resp = cli.complete_with_retry(
        model=model.id,
        messages=messages,
        max_tokens=int(qcfg.get("max_output_tokens") or 256),
        role="qa",
        brief_id=brief_id,
    )
    content = (resp.content or "").strip()
    if (not resp.ok) or _is_empty_content(content):
        # Tiny retry payload
        tiny = {
            "pair_id": brief.pair_id,
            "suite_ok": True,
            "main_py": (extract_main_py(full_text) or "")[:600],
            "ask": "JSON scores+verdict within suite scope",
        }
        resp = cli.complete_with_retry(
            model=model.id,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": json.dumps(tiny, separators=(",", ":"))},
            ],
            max_tokens=int(qcfg.get("max_output_tokens") or 256),
            role="qa",
            brief_id=brief_id,
        )
        content = (resp.content or "").strip()
    if (not resp.ok) or _is_empty_content(content):
        return {
            "ran": True,
            "model": model.id,
            "error": f"{resp.error_class or 'empty'}: {(resp.detail or content)[:200]}",
            "verdict": "needs_edit",
            "scores": {k: 0 for k in _score_keys()},
            "reasons": ["judge_api_error"],
            "edit_brief": [],
            "retry_judge": True,
            "raw_preview": content[:200],
        }
    try:
        parsed = _parse_json_response(content)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "ran": True,
            "model": resp.model,
            "error": str(exc),
            "verdict": "needs_edit",
            "scores": {k: 0 for k in _score_keys()},
            "reasons": ["unparseable_judge_json"],
            "edit_brief": [],
            "retry_judge": True,
            "raw_preview": content[:300],
        }

    scores = {k: _clamp_score((parsed.get("scores") or {}).get(k)) for k in _score_keys()}
    llm_verdict = str(parsed.get("verdict") or "needs_edit").strip().lower()
    if llm_verdict not in {"approve", "needs_edit", "reject"}:
        llm_verdict = "needs_edit"
    ruled = _apply_judge_rules(scores, _min_score(cfg))
    if llm_verdict == "reject":
        verdict: QualityVerdict = "reject"
    elif ruled == "approve":
        verdict = "approve"
    else:
        verdict = "needs_edit"
    return {
        "ran": True,
        "model": resp.model,
        "verdict": verdict,
        "llm_verdict": llm_verdict,
        "scores": scores,
        "reasons": _normalize_str_list(parsed.get("reasons")),
        "edit_brief": _normalize_str_list(parsed.get("edit_brief") or parsed.get("fix_hints")),
    }


def _call_edit(
    *,
    cli: OpenRouterCuratorClient,
    brief: CuratorBrief,
    full_text: str,
    edit_brief: list[str],
    suite_failures: list[str],
) -> tuple[str, str]:
    cfg = cli.config
    retry_cfg = cfg.get("retry") or {}
    max_per_model = int(retry_cfg.get("max_attempts_per_model") or 2)
    max_tokens = int(_quality_cfg(cfg).get("max_gen_tokens") or 1200)
    models = [m.id for m in generation_models(cfg)]
    main_py = extract_main_py(full_text) or ""
    tail = tail_after_prefix(full_text, brief.prefix_lock.prefix_text)[:200]
    hints = "; ".join(_normalize_str_list(edit_brief, limit=3)) or "improve edge handling within suite scope"
    fails = "; ".join(_normalize_str_list(suite_failures, limit=3)) or "none"
    user = (
        f"PREFIX_LOCKED:\n{brief.prefix_lock.prefix_text}\n\n"
        f"NARRATIVE:\n{tail}\n\n"
        f"MAIN_PY:\n```python\n{main_py[:1100]}\n```\n\n"
        f"EDIT: {hints}\n"
        f"SUITE_FAILS: {fails}\n"
        f"CONSTRAINTS: {curator_prompt_constraints(brief.pair_id)[:300]}\n"
        "Return suffix only (narrative + aider main.py). Short stdlib. No tools."
    )
    messages = [
        {"role": "system", "content": EDIT_SYSTEM},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for model in models:
        for _ in range(max_per_model):
            resp = cli.complete_with_retry(
                model=model, messages=messages, max_tokens=max_tokens, role="generation"
            )
            if resp.ok and "```" in resp.content and len(resp.content.strip()) > 60:
                return _strip_fences(resp.content), resp.model
            last_error = f"{model}: {resp.error_class}"
    try:
        resp = cli.complete_generation_last_resort(messages=messages, max_tokens=max_tokens)
        if resp.ok and len(resp.content.strip()) > 40:
            return _strip_fences(resp.content), resp.model
    except (PermissionError, BudgetExhausted) as exc:
        last_error = str(exc)
    return "", last_error


def run_quality_loop(
    *,
    brief: CuratorBrief,
    full_text: str,
    gate_report: dict[str, Any],
    demo_check: dict[str, Any],
    client: OpenRouterCuratorClient | None = None,
    brief_id: str | None = None,
    suite_passed: bool = True,
    suite_detail: str = "",
    judge_calls_so_far: int = 0,
    edits_so_far: int = 0,
) -> dict[str, Any]:
    """Single L5 step: proof check, one judge call, optional one edit."""
    cli = client or OpenRouterCuratorClient()
    cfg = cli.config
    qcfg = _quality_cfg(cfg)
    max_judge = int(qcfg.get("max_judge_calls_per_brief") or 3)
    max_edits = int(qcfg.get("max_edits_per_brief") or 2)
    bid = brief_id or brief.brief_id

    ok, issues = mechanical_qa_preflight(brief=brief, full_text=full_text)
    if not ok:
        return {
            "verdict": "revise",
            "text": full_text,
            "judge": {"ran": False},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls_so_far,
            "fix_hints": issues[:5],
            "preflight_failed": True,
        }

    if not suite_passed:
        return {
            "verdict": "revise",
            "text": full_text,
            "judge": {"ran": False},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls_so_far,
            "fix_hints": [f"quality_suite: {suite_detail}"],
            "llm_skipped": False,
        }

    proof = run_invariant_proof(brief=brief, full_text=full_text)
    main_py = extract_main_py(full_text) or ""
    suite = run_quality_suite(main_py=main_py, pair_id=brief.pair_id, require_configured=True)
    if not suite.pass_:
        return {
            "verdict": "revise",
            "text": full_text,
            "judge": {"ran": False},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls_so_far,
            "fix_hints": [f"RUN_quality_suite: {suite.detail}"],
            "proof": proof,
            "llm_skipped": False,
        }

    if not proof.get("pass"):
        return {
            "verdict": "revise",
            "text": full_text,
            "judge": {"ran": False},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls_so_far,
            "fix_hints": list(proof.get("fix_hints") or [])[:5],
            "proof": proof,
            "llm_skipped": False,
        }

    if judge_calls_so_far >= max_judge:
        return {
            "verdict": "reject",
            "text": full_text,
            "judge": {"ran": False, "reasons": ["judge_budget_exhausted"]},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls_so_far,
            "proof": proof,
            "llm_skipped": False,
        }

    judge = _call_judge(
        cli=cli,
        brief=brief,
        full_text=full_text,
        gate_report=gate_report,
        demo_check=demo_check,
        proof=proof,
        suite_detail=suite_detail or suite.detail,
        brief_id=bid,
    )
    judge_calls = judge_calls_so_far + 1

    if judge.get("retry_judge") and judge_calls < max_judge:
        judge = _call_judge(
            cli=cli,
            brief=brief,
            full_text=full_text,
            gate_report=gate_report,
            demo_check=demo_check,
            proof=proof,
            suite_detail=suite_detail or suite.detail,
            brief_id=bid,
        )
        judge_calls += 1

    verdict = str(judge.get("verdict") or "reject")
    if judge.get("retry_judge") and verdict == "needs_edit":
        return {
            "verdict": "reject",
            "text": full_text,
            "judge": judge,
            "scores": judge.get("scores") or {},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls,
            "proof": proof,
            "llm_skipped": False,
            "reasons": list(judge.get("reasons") or []) + ["judge_unparseable_after_retry"],
            "fix_hints": [],
        }

    if verdict == "approve":
        return {
            "verdict": "approve",
            "text": full_text,
            "judge": judge,
            "scores": judge.get("scores") or {},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls,
            "proof": proof,
            "llm_skipped": False,
            "reasons": list(judge.get("reasons") or []),
            "fix_hints": [],
        }

    if verdict == "reject":
        return {
            "verdict": "reject",
            "text": full_text,
            "judge": judge,
            "scores": judge.get("scores") or {},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls,
            "proof": proof,
            "llm_skipped": False,
            "reasons": list(judge.get("reasons") or []),
            "fix_hints": list(judge.get("edit_brief") or []),
        }

    if edits_so_far >= max_edits:
        return {
            "verdict": "reject",
            "text": full_text,
            "judge": judge,
            "scores": judge.get("scores") or {},
            "edit_count": edits_so_far,
            "judge_calls": judge_calls,
            "proof": proof,
            "llm_skipped": False,
            "reasons": (judge.get("reasons") or []) + ["edit_budget_exhausted"],
            "fix_hints": list(judge.get("edit_brief") or []),
        }

    new_suffix, edit_model = _call_edit(
        cli=cli,
        brief=brief,
        full_text=full_text,
        edit_brief=list(judge.get("edit_brief") or judge.get("reasons") or []),
        suite_failures=[],
    )
    if not new_suffix:
        return {
            "verdict": "reject",
            "text": full_text,
            "judge": judge,
            "edit_count": edits_so_far,
            "judge_calls": judge_calls,
            "proof": proof,
            "llm_skipped": False,
            "error": edit_model,
            "fix_hints": list(judge.get("edit_brief") or []),
        }

    prefix = brief.prefix_lock.prefix_text
    if new_suffix.startswith(prefix):
        new_text = new_suffix
    else:
        new_text = prefix + new_suffix
    return {
        "verdict": "needs_edit",
        "text": new_text,
        "judge": judge,
        "edit_count": edits_so_far + 1,
        "judge_calls": judge_calls,
        "edit_model": edit_model,
        "proof": proof,
        "llm_skipped": False,
        "fix_hints": list(judge.get("edit_brief") or []),
    }

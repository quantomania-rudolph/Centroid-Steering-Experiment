"""Calm-tail curation orchestrator — revise loop, demo oracle, QA budget, attempt ledger."""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py, tail_after_prefix  # noqa: E402
from calm_curator_demo_check import (  # noqa: E402
    check_invariants,
    check_no_hardcode,
    run_demo_check,
    run_demo_stress,
    run_quality_suite,
)
from calm_curator_quality_loop import run_quality_loop  # noqa: E402
from unicaa01_oracle_invariants import (  # noqa: E402
    curator_prompt_constraints,
    sanitize_bad_tail_preview,
)
from calm_curator_schemas import CuratorBrief  # noqa: E402
from divergence_support import get_qwen_tokenizer  # noqa: E402
from jlens_qwen.caa.calm_tail import build_calm_tail_from_file  # noqa: E402
from openrouter_budget import BudgetExhausted, QaBudgetLedger  # noqa: E402
from openrouter_curator_client import (  # noqa: E402
    OpenRouterCuratorClient,
    generation_models,
    load_curator_config,
)
from validate_aider_listing import validate_aider_listing  # noqa: E402
from validate_calm_curator_output import brief_from_dict, validate_calm_curator_output  # noqa: E402
from validate_narrative_purity import narrative_all_pass, validate_narrative_purity  # noqa: E402

PROMPT_TEMPLATE = (ROOT / "research" / "calm_curator_prompt.md").read_text(encoding="utf-8")
STAGING_ROOT = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_staging"

PIPELINE_VERSION = 5

ATTEMPT_BUDGET = 4
HARD_FAIL_GATES = frozenset(
    {
        "CT_repair",
        "CT_sanitize_reassemble",
        "CT_build_spec",
        "NM_d_coherence",
        "CT_file_roundtrip",
        "P1_prefix_match",
        "ASM_no_bridge_on_admit",
        "SAN_token_stable",
        "CD_unrecoverable",
    }
)
FIXABLE_GATES = frozenset(
    {
        "CT_assemble",
        "SY_ast_parse",
        "QA_MAIN_PY_PRESENT",
        "RUN_demo_oracle",
        "RUN_demo_stress",
        "SEM_invariants",
        "SEM_no_hardcode",
        "NARR_no_bridge_catalog",
        "NARR_no_cross_task_phrases",
        "NARR_tail_stub_distress_zero",
        "NARR_continuation_grammar",
        "NARR_balanced_backticks",
        "FMT_aider_path_outside_fence",
        "FMT_no_path_comment_in_fence",
        "RUN_quality_suite",
        "CD_diverges_at_d",
        "PAIR_validate",
    }
)


@dataclass
class ReviseContext:
    failed_checks: list[dict[str, Any]]
    demo_actual_repr: str
    demo_expected_repr: str
    qa_fix_hints: list[str]
    prefix_text: str
    bad_token_at_d_text: str
    expect_repr: str

    def to_prompt_block(self) -> str:
        gates = ", ".join(c["id"] for c in self.failed_checks)
        gate_details = "\n".join(
            f"- {c['id']}: {c.get('detail', '')}" for c in self.failed_checks if not c.get("pass", True)
        )
        hints = "\n".join(f"- {h}" for h in self.qa_fix_hints) or "- fix failed gates"
        return (
            f"REVISION REQUIRED — failed gates: {gates}\n"
            f"Gate details:\n{gate_details or '- see failed gates'}\n"
            f"demo() expected repr: {self.demo_expected_repr}; got: {self.demo_actual_repr or 'N/A'}\n"
            f"Your suffix must make demo() return {self.expect_repr}\n"
            f"Do NOT repeat NotImplementedError at divergence token d.\n"
            f"Prefix is locked — suffix only after:\n{self.prefix_text}\n"
            f"Bad token at d was: {self.bad_token_at_d_text!r}\n"
            f"Fix hints:\n{hints}"
        )


@dataclass
class CurationResult:
    brief_id: str
    status: str
    model: str
    gate_report: dict[str, Any]
    qa_report: dict[str, Any]
    calm_tail_path: str
    error: str = ""
    attempts: dict[str, Any] = field(default_factory=dict)
    qa_calls: int = 0
    bridge_used: bool = False
    sanitized: bool = False
    qa_skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "brief_id": self.brief_id,
            "status": self.status,
            "model": self.model,
            "gate_report": self.gate_report,
            "qa_report": self.qa_report,
            "calm_tail_path": self.calm_tail_path,
            "error": self.error,
            "attempts": self.attempts,
            "qa_calls": self.qa_calls,
            "bridge_used": self.bridge_used,
            "sanitized": self.sanitized,
            "qa_skipped": self.qa_skipped,
        }


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


def _revise_hints_from_failed(failed: list[dict[str, Any]], pair_id: str) -> list[str]:
    hints: list[str] = []
    for c in failed:
        gid = str(c.get("id") or "")
        if gid in (
            "SY_ast_parse",
            "QA_MAIN_PY_PRESENT",
            "FMT_aider_path_outside_fence",
            "FMT_no_path_comment_in_fence",
        ):
            hints.append(
                "Aider listing MUST be path OUTSIDE the fence, then fenced code — never "
                "`# path/main.py` inside the fence. Example:\n"
                "unicaa01_p06_panic/main.py\n```python\ndef parse_json(s): ...\ndef demo(): ...\n```"
            )
        if gid == "SEM_invariants" and pair_id == "p06_json_parse":
            hints.append(
                "Do NOT import or use stdlib json — hand-parse minimal JSON array subset only "
                "(e.g. split on commas inside [..]); demo() must return [1, 2]"
            )
        if gid == "NARR_no_bridge_catalog":
            hints.append("Remove bridge-catalog phrases from narrative; start with task-specific fix line")
        if gid == "NARR_no_cross_task_phrases":
            hints.append("Remove scheduler/threading phrases — this is not a scheduler task")
        if gid == "NARR_continuation_grammar":
            hints.append(
                "Narrative must continue the locked sentence naturally — nowhere in the "
                "first prose block use prompt-paste imperatives (Implement/Write/Create/Edit/Add/Build/Fix the). "
                "Close any open backtick from the prefix (e.g. stub name`)."
            )
        if gid == "NARR_balanced_backticks":
            hints.append(
                "Prefix ends with an open backtick — START the suffix by closing it. "
                "Example suffix start: stub`.  (word then closing tick) then finish the sentence. "
                "Path outside fence next. Do NOT start with Not/NotImplemented."
            )
        if gid == "RUN_quality_suite":
            hints.append(
                f"Quality suite failed: {c.get('detail', '')} — handle [], invalid input, "
                "and bracket validation without import json"
            )
        if gid == "CT_assemble" and "CD divergence" in str(c.get("detail") or ""):
            hints.append(
                "First token after prefix must NOT be the panic token "
                f"({str(c.get('detail') or '').split('bad_text=')[-1]}). "
                "If prefix ends with open `, close with a safe word first "
                "(suffix start like: stub`.) — never Not/NotImplemented."
            )
    return hints


def _gate_dict(gate_id: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"id": gate_id, "pass": passed, "detail": detail}


def _merge_gate_checks(base: dict[str, Any], extra: list[dict[str, Any]]) -> dict[str, Any]:
    checks = list(base.get("checks") or []) + extra
    return {**base, "checks": checks, "pass": all(c.get("pass") for c in checks)}


def _failed_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in report.get("checks") or [] if not c.get("pass")]


def _is_hard_fail(failed: list[dict[str, Any]], *, cd_unrecoverable: bool) -> bool:
    if cd_unrecoverable:
        return True
    ids = {str(c.get("id") or "") for c in failed}
    if ids & HARD_FAIL_GATES:
        return True
    for gate_id in ids:
        if gate_id.startswith("CT_") and gate_id not in ("CT_repair", "CT_assemble"):
            return True
    return False


def _is_fixable(failed: list[dict[str, Any]]) -> bool:
    if not failed:
        return False
    ids = {str(c.get("id") or "") for c in failed}
    if ids <= FIXABLE_GATES or any(g.startswith("FORBID_") for g in ids) or any(
        g.startswith("SY_") or g.startswith("T") for g in ids
    ):
        return True
    return any(g in FIXABLE_GATES for g in ids)


def _run_l1_l4b(
    *,
    brief: CuratorBrief,
    gate_dict: dict[str, Any],
    canonical_text: str,
    main_py: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """L1 demo through L4b quality suite — returns merged gate_dict, failed, demo_result."""
    failed: list[dict[str, Any]] = []
    if not main_py:
        demo_result = {"pass": False, "actual_repr": "", "expected_repr": "", "error": "no main.py"}
        if brief.demo_oracle:
            failed.append(_gate_dict("QA_MAIN_PY_PRESENT", False, "no main.py"))
        return gate_dict, failed, demo_result

    demo_result = (
        run_demo_check(main_py=main_py, pair_id=brief.pair_id).to_dict()
        if brief.demo_oracle
        else {"pass": True, "actual_repr": "", "expected_repr": "", "error": ""}
    )
    demo_gate = _gate_dict("RUN_demo_oracle", bool(demo_result.get("pass")), demo_result.get("error", "") or "ok")
    if not demo_result.get("pass"):
        failed.append(demo_gate)
    gate_dict = _merge_gate_checks(gate_dict, [demo_gate])

    stress_result = run_demo_stress(main_py=main_py, pair_id=brief.pair_id)
    if not stress_result.pass_:
        failed.append(stress_result.to_dict())
    gate_dict = _merge_gate_checks(gate_dict, [stress_result.to_dict()])

    inv_result = check_invariants(main_py=main_py, pair_id=brief.pair_id)
    if not inv_result.pass_:
        failed.append(inv_result.to_dict())
    gate_dict = _merge_gate_checks(gate_dict, [inv_result.to_dict()])

    hc_result = check_no_hardcode(main_py=main_py, pair_id=brief.pair_id)
    if not hc_result.pass_:
        failed.append(hc_result.to_dict())
    gate_dict = _merge_gate_checks(gate_dict, [hc_result.to_dict()])

    narr_results = validate_narrative_purity(
        full_text=canonical_text,
        prefix_text=brief.prefix_lock.prefix_text,
        pair_id=brief.pair_id,
    )
    for nr in narr_results:
        if not nr.pass_:
            failed.append(nr.to_dict())
    gate_dict = _merge_gate_checks(gate_dict, [r.to_dict() for r in narr_results])

    listing_results = validate_aider_listing(
        full_text=canonical_text,
        task_dir=brief.output_contract.task_dir,
    )
    for lr in listing_results:
        if not lr.pass_:
            failed.append(lr.to_dict())
    gate_dict = _merge_gate_checks(gate_dict, [r.to_dict() for r in listing_results])

    suite_result = run_quality_suite(main_py=main_py, pair_id=brief.pair_id, require_configured=True)
    if not suite_result.pass_:
        failed.append(suite_result.to_dict())
    gate_dict = _merge_gate_checks(gate_dict, [suite_result.to_dict()])

    return gate_dict, failed, demo_result


def _build_messages(brief: CuratorBrief, revise: ReviseContext | None = None) -> list[dict[str, str]]:
    pl = brief.prefix_lock
    dc = brief.divergence_context
    expect = brief.demo_oracle.expect_repr if brief.demo_oracle else ""
    constraints = curator_prompt_constraints(brief.pair_id)
    bad_preview = sanitize_bad_tail_preview(brief.pair_id, dc.bad_tail_summary)
    system = PROMPT_TEMPLATE.format(
        prefix_text=pl.prefix_text,
        task_dir=brief.output_contract.task_dir,
        task_summary=brief.task_summary,
        success_criteria=brief.success_criteria,
        panic_starter_main_py=brief.panic_starter_main_py[:1400],
        d=pl.d,
        bad_token_at_d=dc.bad_token_at_d,
        bad_token_at_d_text=dc.bad_token_at_d_text,
        bad_tail_summary=bad_preview,
        expect_repr=expect,
        invariant_constraints=constraints,
    )
    user = (
        f"Continue the assistant completion after this exact prefix (suffix only):\n\n"
        f"{pl.prefix_text}"
    )
    if pl.prefix_text.rstrip().endswith("`"):
        user += (
            "\n\nIMPORTANT: the locked prefix ends with an open backtick. "
            "Your suffix MUST begin by closing it (e.g. stub` or encode`) then finish one short "
            "sentence. Do not start with Not/NotImplemented. Do not use Implement/Write/Create. "
            "Then put the path line outside the fence before ```python."
        )
    if revise is not None:
        user += "\n\n" + revise.to_prompt_block()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_attempt_ledger(brief_id: str, attempts: list[dict[str, Any]]) -> Path:
    staging = STAGING_ROOT / brief_id
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / "attempts.json"
    path.write_text(json.dumps({"attempts": attempts}, indent=2), encoding="utf-8")
    return path


def _generate_suffix(
    *,
    brief: CuratorBrief,
    cli: OpenRouterCuratorClient,
    messages: list[dict[str, str]],
    prefer_paid: bool = False,
) -> tuple[str, str, str]:
    """Generate suffix from roster order (Flash first). prefer_paid kept for call-site compat."""
    _ = prefer_paid
    cfg = cli.config
    retry_cfg = cfg.get("retry") or {}
    max_per_model = int(retry_cfg.get("max_attempts_per_model") or 2)
    max_tokens = int((cfg.get("quality") or {}).get("max_gen_tokens") or 1200)
    last_error = ""
    models = [m.id for m in generation_models(cfg)]
    for model in models:
        for _ in range(max_per_model):
            resp = cli.complete_with_retry(
                model=model, messages=messages, max_tokens=max_tokens, role="generation"
            )
            if resp.ok and "```" in resp.content and len(resp.content.strip()) > 80:
                return _strip_fences(resp.content), model, ""
            last_error = f"{model}: weak_or_{resp.error_class} {(resp.detail or '')[:80]}"
    try:
        resp = cli.complete_generation_last_resort(messages=messages, max_tokens=max_tokens)
        if resp.ok and len(resp.content.strip()) > 40:
            return _strip_fences(resp.content), resp.model, ""
        last_error = f"last_resort: weak_or_{resp.error_class}"
    except (PermissionError, BudgetExhausted) as exc:
        last_error = str(exc)
    return "", "", last_error


def curate_one_brief(
    brief: CuratorBrief | dict[str, Any],
    *,
    client: OpenRouterCuratorClient | None = None,
    allow_skip_qa: bool = False,
    dry_run: bool = False,
    ledger: QaBudgetLedger | None = None,
) -> CurationResult:
    if isinstance(brief, dict):
        brief = brief_from_dict(brief)

    cli = client or OpenRouterCuratorClient(ledger=ledger)
    cfg = cli.config
    tok = get_qwen_tokenizer()
    budget = ledger or cli.ledger or QaBudgetLedger.from_config(cfg)
    qa_calls = 0
    bridge_used_count = 0
    attempt_records: list[dict[str, Any]] = []
    revise_ctx: ReviseContext | None = None
    used_model = ""
    last_cd_fail = False

    for attempt_index in range(ATTEMPT_BUDGET):
        allow_bridge = (
            attempt_index > 0
            and bridge_used_count < 1
            and last_cd_fail
        )
        messages = _build_messages(brief, revise_ctx)
        suffix, used_model, gen_err = _generate_suffix(
            brief=brief, cli=cli, messages=messages
        )
        if not suffix:
            ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
            return CurationResult(
                brief.brief_id,
                "generation_failed",
                used_model,
                {},
                {},
                "",
                gen_err,
                {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                qa_calls,
            )

        gate = validate_calm_curator_output(
            brief=brief,
            candidate_text=suffix,
            tokenizer=tok,
            model=used_model,
            attempt_repair=False,
            sanitize=False,
            strict_assembly=(attempt_index == 0 and not allow_bridge),
            allow_bridge=allow_bridge,
            attempt_index=attempt_index,
            run_miner_parity=False,
            run_demo_oracle=False,
        )
        gate_dict = gate.to_dict()
        failed = _failed_checks(gate_dict)
        canonical_text = gate_dict.get("canonical_text") or ""
        main_py = extract_main_py(canonical_text)

        if gate.pass_ and main_py:
            gate_dict, failed, demo_result = _run_l1_l4b(
                brief=brief,
                gate_dict=gate_dict,
                canonical_text=canonical_text,
                main_py=main_py,
            )
        elif brief.demo_oracle:
            demo_result = {"pass": False, "actual_repr": "", "expected_repr": "", "error": "no main.py"}
        else:
            demo_result = {"pass": True, "actual_repr": "", "expected_repr": "", "error": ""}

        if gate.bridge_used:
            bridge_used_count += 1
        last_cd_fail = any(
            (c.get("id") == "CD_diverges_at_d" and not c.get("pass"))
            or (
                c.get("id") == "CT_assemble"
                and "CD divergence" in str(c.get("detail") or "")
            )
            for c in failed
        )

        cd_unrecoverable = last_cd_fail and bridge_used_count >= 1 and attempt_index > 0
        if cd_unrecoverable:
            failed.append({"id": "CD_unrecoverable", "pass": False, "detail": "CD still fails after bridge"})

        record = {
            "attempt_index": attempt_index,
            "model": used_model,
            "suffix_sha256": _sha256(suffix),
            "bridge_used": gate.bridge_used,
            "sanitized": gate.sanitized,
            "gate_pass": gate.pass_,
            "failed_checks": failed,
            "demo_result": demo_result,
            "qa_verdict": None,
            "qa_calls_so_far": qa_calls,
        }
        attempt_records.append(record)

        if not gate.pass_ or failed:
            if _is_hard_fail(failed, cd_unrecoverable=cd_unrecoverable):
                staging = Path(brief.output_contract.target_staging_path)
                staging.parent.mkdir(parents=True, exist_ok=True)
                if canonical_text:
                    staging.write_text(canonical_text, encoding="utf-8")
                (staging.parent / "gate_report.json").write_text(json.dumps(gate_dict, indent=2), encoding="utf-8")
                ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
                return CurationResult(
                    brief.brief_id,
                    "gate_failed",
                    used_model,
                    gate_dict,
                    {},
                    "",
                    "hard_fail",
                    {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                    qa_calls,
                    gate.bridge_used,
                    gate.sanitized,
                )
            if _is_fixable(failed) and attempt_index < ATTEMPT_BUDGET - 1:
                revise_ctx = ReviseContext(
                    failed_checks=failed,
                    demo_actual_repr=str(demo_result.get("actual_repr") or ""),
                    demo_expected_repr=str(demo_result.get("expected_repr") or ""),
                    qa_fix_hints=_revise_hints_from_failed(failed, brief.pair_id),
                    prefix_text=brief.prefix_lock.prefix_text,
                    bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                    expect_repr=brief.demo_oracle.expect_repr if brief.demo_oracle else "",
                )
                continue
            staging = Path(brief.output_contract.target_staging_path)
            staging.parent.mkdir(parents=True, exist_ok=True)
            if canonical_text:
                staging.write_text(canonical_text, encoding="utf-8")
            ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
            return CurationResult(
                brief.brief_id,
                "gate_failed",
                used_model,
                gate_dict,
                {},
                "",
                "exhausted_fixable",
                {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                qa_calls,
                gate.bridge_used,
                gate.sanitized,
            )

        qa_report: dict[str, Any] = {"verdict": "approve", "reasons": ["qa_skipped"], "qa_skipped": True}
        quality_edits = 0
        quality_judge_calls = 0
        edit_revise = False
        if not allow_skip_qa:
            suite_passed = not any(c.get("id") == "RUN_quality_suite" and not c.get("pass") for c in failed)
            suite_detail = next(
                (str(c.get("detail") or "") for c in failed if c.get("id") == "RUN_quality_suite"),
                "",
            )
            while True:
                try:
                    loop_result = run_quality_loop(
                        brief=brief,
                        full_text=canonical_text,
                        gate_report=gate_dict,
                        demo_check=demo_result,
                        client=cli,
                        brief_id=brief.brief_id,
                        suite_passed=suite_passed,
                        suite_detail=suite_detail,
                        judge_calls_so_far=quality_judge_calls,
                        edits_so_far=quality_edits,
                    )
                except BudgetExhausted as exc:
                    ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
                    return CurationResult(
                        brief.brief_id,
                        "budget_exhausted",
                        used_model,
                        gate_dict,
                        {},
                        "",
                        str(exc),
                        {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                        budget.qa_calls_for_brief(brief.brief_id),
                        gate.bridge_used,
                        gate.sanitized,
                    )
                quality_judge_calls = int(loop_result.get("judge_calls") or quality_judge_calls)
                quality_edits = int(loop_result.get("edit_count") or quality_edits)
                loop_verdict = str(loop_result.get("verdict") or "reject")

                if loop_verdict == "needs_edit":
                    prior_text = canonical_text
                    edited_text = str(loop_result.get("text") or canonical_text)
                    canonical_text = edited_text
                    suffix = (
                        canonical_text[len(brief.prefix_lock.prefix_text) :]
                        if canonical_text.startswith(brief.prefix_lock.prefix_text)
                        else canonical_text
                    )
                    gate = validate_calm_curator_output(
                        brief=brief,
                        candidate_text=suffix,
                        tokenizer=tok,
                        model=used_model,
                        attempt_repair=False,
                        sanitize=False,
                        strict_assembly=(attempt_index == 0),
                        allow_bridge=False,
                        attempt_index=attempt_index,
                        run_miner_parity=False,
                        run_demo_oracle=False,
                    )
                    gate_dict = gate.to_dict()
                    failed = _failed_checks(gate_dict)
                    main_py = extract_main_py(gate_dict.get("canonical_text") or canonical_text)
                    canonical_text = gate_dict.get("canonical_text") or canonical_text
                    if gate.pass_ and main_py:
                        gate_dict, failed, demo_result = _run_l1_l4b(
                            brief=brief,
                            gate_dict=gate_dict,
                            canonical_text=canonical_text,
                            main_py=main_py,
                        )
                    suite_passed = not any(
                        c.get("id") == "RUN_quality_suite" and not c.get("pass") for c in failed
                    )
                    suite_detail = next(
                        (str(c.get("detail") or "") for c in failed if c.get("id") == "RUN_quality_suite"),
                        "",
                    )
                    if failed:
                        # Discard broken edit — restore prior suite-passing candidate
                        canonical_text = prior_text
                        main_py_prior = extract_main_py(prior_text)
                        if main_py_prior:
                            gate_dict, failed_prior, demo_result = _run_l1_l4b(
                                brief=brief,
                                gate_dict={"pass": True, "checks": list(gate_dict.get("checks") or [])},
                                canonical_text=prior_text,
                                main_py=main_py_prior,
                            )
                            failed = failed_prior
                        suite_passed = not any(
                            c.get("id") == "RUN_quality_suite" and not c.get("pass") for c in failed
                        )
                        suite_detail = next(
                            (str(c.get("detail") or "") for c in failed if c.get("id") == "RUN_quality_suite"),
                            "edit discarded — prior restored",
                        )
                        if failed:
                            if attempt_index < ATTEMPT_BUDGET - 1:
                                revise_ctx = ReviseContext(
                                    failed_checks=failed,
                                    demo_actual_repr=str(demo_result.get("actual_repr") or ""),
                                    demo_expected_repr=str(demo_result.get("expected_repr") or ""),
                                    qa_fix_hints=list(loop_result.get("fix_hints") or [])
                                    + ["Quality edit broke gates — regenerate a correct robust suffix"],
                                    prefix_text=brief.prefix_lock.prefix_text,
                                    bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                                    expect_repr=brief.demo_oracle.expect_repr if brief.demo_oracle else "",
                                )
                                edit_revise = True
                                break
                            loop_verdict = "reject"
                        else:
                            # Prior still good — another judge/edit round if budget remains
                            continue
                    else:
                        continue

                qa_report = {
                    "verdict": "approve" if loop_verdict == "approve" else loop_verdict,
                    "scores": loop_result.get("scores") or {},
                    "reasons": list(loop_result.get("reasons") or []),
                    "fix_hints": list(loop_result.get("fix_hints") or []),
                    "semantic_qa": True,
                    "proof": loop_result.get("proof"),
                    "judge": loop_result.get("judge") or {},
                    "judge.ran": bool((loop_result.get("judge") or {}).get("ran")),
                    "llm_skipped": False,
                    "edit_count": quality_edits,
                    "judge_calls": quality_judge_calls,
                    "quality_loop": loop_result,
                    "pipeline_version": PIPELINE_VERSION,
                }
                break

            if edit_revise:
                continue

            qa_calls = budget.qa_calls_for_brief(brief.brief_id)
            record["qa_verdict"] = qa_report.get("verdict")
            record["qa_calls_so_far"] = qa_calls

            if qa_report.get("verdict") == "revise" and attempt_index < ATTEMPT_BUDGET - 1:
                revise_ctx = ReviseContext(
                    failed_checks=failed,
                    demo_actual_repr=str(demo_result.get("actual_repr") or ""),
                    demo_expected_repr=str(demo_result.get("expected_repr") or ""),
                    qa_fix_hints=list(qa_report.get("fix_hints") or []),
                    prefix_text=brief.prefix_lock.prefix_text,
                    bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                    expect_repr=brief.demo_oracle.expect_repr if brief.demo_oracle else "",
                )
                continue
            if (
                qa_report.get("verdict") == "reject"
                and attempt_index < ATTEMPT_BUDGET - 1
                and any(
                    r in {"unparseable_judge_json", "judge_api_error"}
                    for r in (qa_report.get("reasons") or [])
                )
            ):
                revise_ctx = ReviseContext(
                    failed_checks=[],
                    demo_actual_repr=str(demo_result.get("actual_repr") or ""),
                    demo_expected_repr=str(demo_result.get("expected_repr") or ""),
                    qa_fix_hints=list(qa_report.get("fix_hints") or [])
                    + ["Keep suite-passing parser; improve narrative and edge clarity"],
                    prefix_text=brief.prefix_lock.prefix_text,
                    bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                    expect_repr=brief.demo_oracle.expect_repr if brief.demo_oracle else "",
                )
                continue
            if qa_report.get("verdict") not in ("approve",):
                staging = Path(brief.output_contract.target_staging_path)
                staging.parent.mkdir(parents=True, exist_ok=True)
                staging.write_text(canonical_text, encoding="utf-8")
                (staging.parent / "qa_report.json").write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
                ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
                return CurationResult(
                    brief.brief_id,
                    "qa_rejected",
                    used_model,
                    gate_dict,
                    qa_report,
                    "",
                    "",
                    {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                    qa_calls,
                    gate.bridge_used,
                    gate.sanitized,
                )

        parity_gate = validate_calm_curator_output(
            brief=brief,
            candidate_text=canonical_text,
            tokenizer=tok,
            model=used_model,
            attempt_repair=False,
            sanitize=False,
            parity_only=True,
            run_miner_parity=True,
            run_demo_oracle=False,
        )
        if not parity_gate.pass_:
            parity_failed = _failed_checks(parity_gate.to_dict())
            gate_dict = _merge_gate_checks(gate_dict, parity_gate.to_dict().get("checks") or [])
            record["gate_pass"] = False
            record["failed_checks"] = parity_failed
            if _is_fixable(parity_failed) and attempt_index < ATTEMPT_BUDGET - 1:
                revise_ctx = ReviseContext(
                    failed_checks=parity_failed,
                    demo_actual_repr=str(demo_result.get("actual_repr") or ""),
                    demo_expected_repr=str(demo_result.get("expected_repr") or ""),
                    qa_fix_hints=["Fix miner parity failures"],
                    prefix_text=brief.prefix_lock.prefix_text,
                    bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                    expect_repr=brief.demo_oracle.expect_repr if brief.demo_oracle else "",
                )
                continue
            ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
            return CurationResult(
                brief.brief_id,
                "gate_failed",
                used_model,
                gate_dict,
                qa_report,
                "",
                "PAIR_validate failed",
                {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                qa_calls,
                gate.bridge_used,
                gate.sanitized,
            )
        gate_dict = _merge_gate_checks(gate_dict, parity_gate.to_dict().get("checks") or [])

        if gate.repaired or gate.bridge_used or gate.sanitized:
            if attempt_index < ATTEMPT_BUDGET - 1:
                revise_ctx = ReviseContext(
                    failed_checks=[{"id": "ASM_no_bridge_on_admit", "pass": False, "detail": "bridge/sanitize/repair used"}],
                    demo_actual_repr=str(demo_result.get("actual_repr") or ""),
                    demo_expected_repr=str(demo_result.get("expected_repr") or ""),
                    qa_fix_hints=["Regenerate without bridge, sanitize, or repair"],
                    prefix_text=brief.prefix_lock.prefix_text,
                    bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                    expect_repr=brief.demo_oracle.expect_repr if brief.demo_oracle else "",
                )
                continue
            ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
            return CurationResult(
                brief.brief_id,
                "admit_invariant_failed",
                used_model,
                gate_dict,
                qa_report,
                "",
                "repaired/bridge/sanitize on admit",
                {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                qa_calls,
                gate.bridge_used,
                gate.sanitized,
                allow_skip_qa,
            )

        admitted = Path(brief.output_contract.target_admitted_path)
        if allow_skip_qa and not dry_run:
            ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
            return CurationResult(
                brief.brief_id,
                "qa_skipped_not_admitted",
                used_model,
                gate_dict,
                qa_report,
                "",
                "v4 requires judge — refuse admit when --allow-skip-qa",
                {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                qa_calls,
                False,
                False,
                True,
            )
        if not dry_run:
            admitted.parent.mkdir(parents=True, exist_ok=True)
            admitted.write_text(canonical_text, encoding="utf-8")
            try:
                build_calm_tail_from_file(
                    prefix_token_ids=brief.prefix_lock.prefix_token_ids,
                    d=brief.prefix_lock.d,
                    calm_completion_path=admitted,
                    tokenizer=tok,
                )
            except (ValueError, FileNotFoundError) as exc:
                ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
                return CurationResult(
                    brief.brief_id,
                    "roundtrip_failed",
                    used_model,
                    gate_dict,
                    qa_report,
                    "",
                    str(exc),
                    {"total": len(attempt_records), "ledger_path": str(ledger_path)},
                    qa_calls,
                    False,
                    False,
                    allow_skip_qa,
                )
            gate_flags = {
                c["id"]: c.get("pass")
                for c in gate_dict.get("checks") or []
                if c.get("id") in {
                    "RUN_demo_oracle",
                    "RUN_demo_stress",
                    "RUN_quality_suite",
                    "SEM_invariants",
                    "SEM_no_hardcode",
                    "NARR_no_bridge_catalog",
                    "NARR_no_cross_task_phrases",
                    "NARR_tail_stub_distress_zero",
                    "NARR_continuation_grammar",
                    "NARR_balanced_backticks",
                    "FMT_aider_path_outside_fence",
                    "FMT_no_path_comment_in_fence",
                    "PAIR_validate",
                }
            }
            meta = {
                "brief_id": brief.brief_id,
                "model": used_model,
                "status": "admitted",
                "pipeline_version": PIPELINE_VERSION,
                "gate": gate_dict,
                "gates": gate_flags,
                "qa": qa_report,
                "semantic_qa": qa_report,
                "quality_suite": {
                    "pass": gate_flags.get("RUN_quality_suite"),
                    "edit_count": quality_edits,
                },
                "judge": qa_report.get("judge") or {},
                "demo": demo_result,
                "repaired": gate.repaired,
                "bridge_used": gate.bridge_used,
                "sanitized": gate.sanitized,
                "attempt_repair": False,
                "qa_calls": qa_calls,
                "qa_skipped": allow_skip_qa,
                "narrative_prefix_80": tail_after_prefix(
                    canonical_text, brief.prefix_lock.prefix_text
                )[:80],
            }
            (admitted.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
        return CurationResult(
            brief.brief_id,
            "admitted",
            used_model,
            gate_dict,
            qa_report,
            str(admitted),
            "",
            {"total": len(attempt_records), "ledger_path": str(ledger_path)},
            qa_calls,
            False,
            False,
            allow_skip_qa,
        )

    ledger_path = _write_attempt_ledger(brief.brief_id, attempt_records)
    return CurationResult(
        brief.brief_id,
        "attempts_exhausted",
        used_model,
        {},
        {},
        "",
        "max attempts",
        {"total": len(attempt_records), "ledger_path": str(ledger_path)},
        qa_calls,
    )


def curate_batch(
    briefs: list[CuratorBrief | dict[str, Any]],
    *,
    max_parallel: int | None = None,
    allow_skip_qa: bool = False,
    dry_run: bool = False,
    ledger: QaBudgetLedger | None = None,
) -> list[CurationResult]:
    cfg = load_curator_config()
    parallel = max_parallel or int(cfg.get("max_parallel") or 2)
    shared_ledger = ledger or QaBudgetLedger.from_config(cfg)
    results: list[CurationResult] = []

    if parallel <= 1 or len(briefs) <= 1:
        client = OpenRouterCuratorClient(ledger=shared_ledger)
        for b in briefs:
            results.append(
                curate_one_brief(b, client=client, allow_skip_qa=allow_skip_qa, dry_run=dry_run, ledger=shared_ledger)
            )
        return results

    def _run_one(b: CuratorBrief | dict[str, Any]) -> CurationResult:
        thread_client = OpenRouterCuratorClient(ledger=shared_ledger)
        return curate_one_brief(
            b,
            client=thread_client,
            allow_skip_qa=allow_skip_qa,
            dry_run=dry_run,
            ledger=shared_ledger,
        )

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(_run_one, b): b for b in briefs}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results

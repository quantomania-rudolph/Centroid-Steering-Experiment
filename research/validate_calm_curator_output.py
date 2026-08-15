"""Mechanical gates for calm curator output — CT, CD, SY, T, P, NM miner-parity."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import assemble_calm_completion, extract_main_py  # noqa: E402
from calm_curator_schemas import (  # noqa: E402
    CuratorBrief,
    CuratorGateReport,
    DemoOracle,
    DivergenceContext,
    GateCheck,
    OutputContract,
    PrefixLock,
)
from calm_curator_demo_check import run_demo_check  # noqa: E402
from divergence_support import get_qwen_tokenizer, resolve_assistant_text  # noqa: E402
from jlens_qwen.caa.calm_tail import build_calm_tail_from_ids, validate_prefix_lock  # noqa: E402
from jlens_qwen.caa.method_b_databus import build_method_b_record  # noqa: E402
from jlens_qwen.caa.miner_pairing import (  # noqa: E402
    build_clean_chaos_from_token_ids,
    check_d_coherence,
    extract_d_from_lcp,
)
from prefix_divergence_tracer import trace_invocation  # noqa: E402
from repair_calm_tail import repair_prefix_lock, sanitize_full_completion  # noqa: E402
from validate_caa_pair import validate_pair  # noqa: E402


def brief_from_dict(data: dict[str, Any]) -> CuratorBrief:
    pl = data["prefix_lock"]
    dc = data["divergence_context"]
    oc = data["output_contract"]
    demo_raw = data.get("demo_oracle")
    demo_oracle: DemoOracle | None = None
    if demo_raw:
        demo_oracle = DemoOracle(
            pair_id=str(demo_raw["pair_id"]),
            check_mode=demo_raw["check_mode"],  # type: ignore[arg-type]
            expect_repr=str(demo_raw["expect_repr"]),
        )
    return CuratorBrief(
        brief_version=int(data.get("brief_version") or 1),
        brief_id=str(data["brief_id"]),
        task_slug=str(data["task_slug"]),
        pair_id=str(data["pair_id"]),
        invocation_id=str(data["invocation_id"]),
        domain=str(data["domain"]),
        user_message=str(data["user_message"]),
        panic_starter_main_py=str(data["panic_starter_main_py"]),
        prefix_lock=PrefixLock(
            d=int(pl["d"]),
            prefix_token_ids=list(pl["prefix_token_ids"]),
            prefix_text=str(pl["prefix_text"]),
            immutable=bool(pl.get("immutable", True)),
        ),
        divergence_context=DivergenceContext(
            bad_token_at_d=dc.get("bad_token_at_d"),
            bad_token_at_d_text=str(dc.get("bad_token_at_d_text") or ""),
            bad_tail_summary=str(dc.get("bad_tail_summary") or ""),
            forbidden_patterns=list(dc.get("forbidden_patterns") or []),
        ),
        output_contract=OutputContract(
            format=str(oc.get("format") or "aider_listing"),
            target_staging_path=str(oc["target_staging_path"]),
            target_admitted_path=str(oc["target_admitted_path"]),
            task_dir=str(oc.get("task_dir") or data["task_slug"]),
        ),
        success_criteria=str(data.get("success_criteria") or ""),
        task_summary=str(data.get("task_summary") or ""),
        provenance=dict(data.get("provenance") or {}),
        tracer_confidence=str(data.get("tracer_confidence") or "medium"),
        demo_oracle=demo_oracle,
    )


def _load_bad_context(brief: CuratorBrief) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    src = Path(brief.provenance.get("source_experiment_json") or "")
    if not src.is_file():
        raise FileNotFoundError(f"missing source experiment: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    inv = next(
        (i for i in data.get("invocations") or [] if str(i.get("id")) == brief.invocation_id),
        None,
    )
    if inv is None:
        raise ValueError(f"invocation {brief.invocation_id} not in {src}")
    return data, inv, data


def validate_calm_curator_output(
    *,
    brief: CuratorBrief | dict[str, Any],
    candidate_text: str,
    tokenizer: Any | None = None,
    model: str = "",
    attempt_repair: bool = True,
    sanitize: bool = True,
    strict_assembly: bool = False,
    allow_bridge: bool = True,
    attempt_index: int = 0,
    run_miner_parity: bool = True,
    run_demo_oracle: bool = True,
    parity_only: bool = False,
) -> CuratorGateReport:
    if isinstance(brief, dict):
        brief = brief_from_dict(brief)

    tok = tokenizer or get_qwen_tokenizer()
    d = brief.prefix_lock.d
    prefix_ids = list(brief.prefix_lock.prefix_token_ids)
    prefix_text = brief.prefix_lock.prefix_text
    checks: list[GateCheck] = []
    repaired = False
    text = candidate_text
    was_sanitized = False
    assembled_bridge_used = False
    main_py = ""
    canonical_text = ""

    if parity_only:
        canonical_text = candidate_text
        calm_ids = tok.encode(canonical_text, add_special_tokens=False)
        if hasattr(calm_ids, "tolist"):
            calm_ids = [int(x) for x in calm_ids.tolist()]
        else:
            calm_ids = [int(x) for x in calm_ids]
        main_py = extract_main_py(canonical_text)
    else:
        if attempt_repair:
            try:
                text, repaired = repair_prefix_lock(
                    prefix_token_ids=prefix_ids,
                    prefix_text=prefix_text,
                    candidate_text=candidate_text,
                    tokenizer=tok,
                    d=d,
                    bad_token_at_d=brief.divergence_context.bad_token_at_d,
                    bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                )
            except ValueError as exc:
                checks.append(GateCheck("CT_repair", False, str(exc)))
                return CuratorGateReport(brief.brief_id, model, False, checks, repaired)

        try:
            assembled = assemble_calm_completion(
                prefix_token_ids=prefix_ids,
                prefix_text=prefix_text,
                suffix_text=text,
                tokenizer=tok,
                d=d,
                bad_token_at_d=brief.divergence_context.bad_token_at_d,
                bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                strict=strict_assembly,
                allow_bridge=allow_bridge,
            )
        except ValueError as exc:
            checks.append(GateCheck("CT_assemble", False, str(exc)))
            return CuratorGateReport(brief.brief_id, model, False, checks, repaired)

        was_sanitized = False
        if sanitize:
            sanitized_text = sanitize_full_completion(assembled.full_text, prefix_text)
            if sanitized_text != assembled.full_text:
                was_sanitized = True
                repaired = True
                try:
                    assembled = assemble_calm_completion(
                        prefix_token_ids=prefix_ids,
                        prefix_text=prefix_text,
                        suffix_text=sanitized_text[len(prefix_text) :],
                        tokenizer=tok,
                        d=d,
                        bad_token_at_d=brief.divergence_context.bad_token_at_d,
                        bad_token_at_d_text=brief.divergence_context.bad_token_at_d_text,
                        allow_bridge=allow_bridge,
                    )
                except ValueError as exc:
                    checks.append(GateCheck("CT_sanitize_reassemble", False, str(exc)))
                    return CuratorGateReport(brief.brief_id, model, False, checks, repaired)

        calm_ids = assembled.full_token_ids
        ct = validate_prefix_lock(calm_ids, prefix_ids, d)
        for row in ct["checks"]:
            checks.append(GateCheck(str(row["id"]), bool(row["pass"]), str(row["detail"])))

        exp_data, inv, _ = _load_bad_context(brief)
        bad_ids = list((inv.get("chaos") or {}).get("token_ids") or [])
        cd_ok = len(calm_ids) > d and len(bad_ids) > d and calm_ids[d] != bad_ids[d]
        checks.append(
            GateCheck(
                "CD_diverges_at_d",
                cd_ok,
                f"calm@{d}={calm_ids[d] if len(calm_ids) > d else None} "
                f"bad@{d}={bad_ids[d] if len(bad_ids) > d else None}",
            )
        )

        checks.append(
            GateCheck(
                "ASM_no_bridge_on_admit",
                not (attempt_index == 0 and assembled.bridge_used),
                f"attempt_index={attempt_index} bridge_used={assembled.bridge_used}",
            )
        )

        main_py = extract_main_py(assembled.full_text)
        sy_ok = bool(main_py)
        sy_detail = "no fenced main.py block"
        if main_py:
            try:
                ast.parse(main_py)
                sy_detail = f"parsed {len(main_py)} chars"
            except SyntaxError as exc:
                sy_ok = False
                sy_detail = str(exc)
        checks.append(GateCheck("SY_ast_parse", sy_ok, sy_detail))
        checks.append(
            GateCheck(
                "QA_MAIN_PY_PRESENT",
                bool(main_py),
                f"extracted_len={len(main_py)}",
            )
        )

        code_scan = main_py if main_py else assembled.suffix_text
        narrative_scan = assembled.suffix_text
        stub_patterns = {"NotImplementedError", "TODO", "FIXME"}
        narrative_word_patterns = {"pip install", "sorry", "cannot", "unable to"}

        def _forbidden_hit(pat: str, scan_text: str, *, word_boundary: bool) -> bool:
            if word_boundary:
                return bool(re.search(rf"\b{re.escape(pat)}\b", scan_text, re.IGNORECASE))
            return pat.lower() in scan_text.lower()

        for pat in brief.divergence_context.forbidden_patterns:
            if pat in stub_patterns:
                hit = _forbidden_hit(pat, code_scan, word_boundary=False)
            elif pat in narrative_word_patterns:
                hit = _forbidden_hit(pat, narrative_scan, word_boundary=True)
            else:
                hit = _forbidden_hit(pat, narrative_scan, word_boundary=False)
            checks.append(GateCheck(f"FORBID_{pat[:12]}", not hit, f"found={hit}"))

        canonical_text = assembled.full_text
        re_ids = tok.encode(canonical_text, add_special_tokens=False)
        if hasattr(re_ids, "tolist"):
            re_ids = [int(x) for x in re_ids.tolist()]
        else:
            re_ids = [int(x) for x in re_ids]
        rt_gate = validate_prefix_lock(re_ids, prefix_ids, d)
        checks.append(
            GateCheck(
                "CT_file_roundtrip",
                bool(rt_gate["pass"]),
                f"canonical_len={len(re_ids)} calm_len={len(calm_ids)}",
            )
        )

        try:
            build_calm_tail_from_ids(
                prefix_token_ids=prefix_ids,
                calm_full_token_ids=calm_ids,
                d=d,
                calm_text=assembled.full_text,
            )
            checks.append(GateCheck("CT_build_spec", True, "ok"))
        except ValueError as exc:
            checks.append(GateCheck("CT_build_spec", False, str(exc)))
            return CuratorGateReport(brief.brief_id, model, False, checks, repaired)

        assembled_bridge_used = assembled.bridge_used

    if run_miner_parity:
        if not parity_only:
            exp_data, inv, _ = _load_bad_context(brief)
            bad_ids = list((inv.get("chaos") or {}).get("token_ids") or [])
        else:
            exp_data, inv, _ = _load_bad_context(brief)
            bad_ids = list((inv.get("chaos") or {}).get("token_ids") or [])

        try:
            calm_spec = build_calm_tail_from_ids(
                prefix_token_ids=prefix_ids,
                calm_full_token_ids=calm_ids,
                d=d,
                calm_text=canonical_text,
            )
        except ValueError as exc:
            checks.append(GateCheck("CT_build_spec", False, str(exc)))
            return CuratorGateReport(brief.brief_id, model, False, checks, repaired)

        d_extract = extract_d_from_lcp(calm_ids, bad_ids)
        d_chk = check_d_coherence(d_extract, d)
        checks.append(GateCheck(d_chk.id, d_chk.passed, d_chk.detail))
        if not d_chk.passed:
            return CuratorGateReport(brief.brief_id, model, False, checks, repaired, "")

        d_spec = d_extract
        trace = trace_invocation(
            inv,
            experiment=str(exp_data.get("experiment") or ""),
            task_slug=str(exp_data.get("task_slug") or ""),
            experiment_data=exp_data,
            use_tokenizer=False,
        )
        bad_text = resolve_assistant_text(inv, exp_data)
        prompt_len = int(inv.get("prompt_tokens") or 0)
        spec = {
            "d": d_spec,
            "prefix_token_ids": list(calm_spec.prefix_token_ids)[:d_spec],
            "calm_tail_token_ids": calm_spec.calm_tail_token_ids,
            "bad_tail_token_ids": bad_ids[d_spec:],
            "calm_tail_text": calm_spec.calm_text,
            "bad_tail_text": bad_text[d_spec:] if bad_text else "",
        }
        calm_chaos = build_clean_chaos_from_token_ids(calm_spec.calm_full_token_ids)
        calm_mb = build_method_b_record(
            task_slug=brief.task_slug,
            experiment=str(exp_data.get("experiment") or ""),
            invocation_id=f"{brief.invocation_id}_calm",
            user_prompt=brief.user_message,
            divergence=trace.to_dict(),
            tail_token_ids=calm_spec.calm_tail_token_ids,
            tail_text=calm_spec.calm_text,
            tail_kind="calm",
            prompt_len=prompt_len,
            fallback_d=d_spec,
            fallback_prefix_token_ids=spec["prefix_token_ids"],
        )
        bad_mb = build_method_b_record(
            task_slug=brief.task_slug,
            experiment=str(exp_data.get("experiment") or ""),
            invocation_id=brief.invocation_id,
            user_prompt=brief.user_message,
            divergence=trace.to_dict(),
            tail_token_ids=bad_ids[d_spec:],
            tail_text=spec["bad_tail_text"],
            tail_kind="observed",
            prompt_len=prompt_len,
        )
        calm_run = {
            "assistant_text": calm_spec.calm_text,
            "tail_text": calm_spec.calm_text,
            "chaos": calm_chaos,
            "invocation": {
                "id": f"{brief.invocation_id}_calm",
                "completion_tokens": len(calm_spec.calm_full_token_ids),
                "assistant": calm_spec.calm_text,
                "chaos": calm_chaos,
            },
            "experiment": str(exp_data.get("experiment") or ""),
            "task_slug": brief.task_slug,
            "method_b_record": calm_mb.to_dict() if calm_mb else None,
        }
        bad_run = {
            "assistant_text": bad_text,
            "chaos": inv.get("chaos") or {},
            "invocation": inv,
            "experiment": str(exp_data.get("experiment") or ""),
            "task_slug": brief.task_slug,
            "method_b_record": bad_mb.to_dict() if bad_mb else None,
        }
        pair = validate_pair(spec=spec, calm_run=calm_run, bad_run=bad_run, prompt_len=prompt_len)
        checks.append(GateCheck("PAIR_validate", bool(pair["pass"]), "miner-parity dry-run"))
        for row in pair.get("pair_checks") or []:
            checks.append(GateCheck(str(row["id"]), bool(row["pass"]), str(row["detail"])[:200]))
        for row in (pair.get("calm") or {}).get("checks") or []:
            checks.append(GateCheck(str(row["id"]), bool(row["pass"]), str(row["detail"])[:200]))
        for row in (pair.get("bad") or {}).get("checks") or []:
            checks.append(GateCheck(str(row["id"]), bool(row["pass"]), str(row["detail"])[:200]))

    if run_demo_oracle and not parity_only:
        if brief.demo_oracle is not None and main_py:
            demo_result = run_demo_check(main_py=main_py, oracle=brief.demo_oracle)
            checks.append(
                GateCheck(
                    "RUN_demo_oracle",
                    demo_result.pass_,
                    demo_result.error or f"actual={demo_result.actual_repr}",
                )
            )
        elif brief.demo_oracle is not None:
            checks.append(GateCheck("RUN_demo_oracle", False, "no main.py for demo check"))

    ok = all(c.pass_ for c in checks)
    calm_path = brief.output_contract.target_admitted_path if ok else ""
    report = CuratorGateReport(
        brief.brief_id,
        model,
        ok,
        checks,
        repaired,
        calm_path if ok else "",
        canonical_text if ok else "",
        assembled_bridge_used,
        was_sanitized,
    )
    return report


def write_gate_aggregate(reports: list[CuratorGateReport], out: Path) -> dict[str, Any]:
    payload = {
        "n_reports": len(reports),
        "n_pass": sum(1 for r in reports if r.pass_),
        "reports": [r.to_dict() for r in reports],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

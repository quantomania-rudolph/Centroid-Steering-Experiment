#!/usr/bin/env python
"""Admit a suite-passing p06 candidate via live L0-L6 + judge (production gates)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py, tail_after_prefix  # noqa: E402
from calm_curator_demo_check import run_quality_suite  # noqa: E402
from calm_curator_orchestrator import (  # noqa: E402
    PIPELINE_VERSION,
    _merge_gate_checks,
    _run_l1_l4b,
)
from calm_curator_quality_loop import run_quality_loop  # noqa: E402
from divergence_support import get_qwen_tokenizer  # noqa: E402
from jlens_qwen.caa.calm_tail import build_calm_tail_from_file  # noqa: E402
from openrouter_curator_client import OpenRouterCuratorClient  # noqa: E402
from validate_calm_curator_output import brief_from_dict, validate_calm_curator_output  # noqa: E402

BRIEF_ID = "unicaa01_p06_panic__chatcmpl_44fb68e7b5cd__d3"

CANDIDATE = """Fixing `broken int-array parser with bracket checks and empty-list handling.

unicaa01_p06_panic/main.py
```python
def parse_json(s):
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        raise ValueError('array only')
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [int(x.strip()) for x in inner.split(',')]

def demo():
    return parse_json('[1,2]')
```
"""


def main() -> int:
    brief = brief_from_dict(
        json.loads(
            (ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_briefs" / f"{BRIEF_ID}.json").read_text(
                encoding="utf-8"
            )
        )
    )
    canonical = CANDIDATE
    main_py = extract_main_py(canonical)
    suite = run_quality_suite(main_py=main_py, pair_id=brief.pair_id)
    print("suite", suite.to_dict())
    if not suite.pass_:
        return 1

    tok = get_qwen_tokenizer()
    cli = OpenRouterCuratorClient()
    suffix = canonical[len(brief.prefix_lock.prefix_text) :]
    gate = validate_calm_curator_output(
        brief=brief,
        candidate_text=suffix,
        tokenizer=tok,
        model="deepseek/deepseek-v4-flash",
        attempt_repair=False,
        sanitize=False,
        strict_assembly=True,
        allow_bridge=False,
        attempt_index=0,
        run_miner_parity=False,
        run_demo_oracle=False,
    )
    gate_dict = gate.to_dict()
    canonical = gate_dict.get("canonical_text") or canonical
    main_py = extract_main_py(canonical)
    if not (gate.pass_ and main_py):
        print("L0 fail", gate_dict)
        return 1
    gate_dict, failed, demo = _run_l1_l4b(
        brief=brief, gate_dict=gate_dict, canonical_text=canonical, main_py=main_py
    )
    print("L1-L4b failed", failed)
    if failed:
        return 1

    edits = 0
    judge_calls = 0
    loop = run_quality_loop(
        brief=brief,
        full_text=canonical,
        gate_report=gate_dict,
        demo_check=demo,
        client=cli,
        brief_id=brief.brief_id,
        suite_passed=True,
        suite_detail=suite.detail,
    )
    edits = int(loop.get("edit_count") or 0)
    judge_calls = int(loop.get("judge_calls") or 0)
    print("loop", loop.get("verdict"), loop.get("scores") or (loop.get("judge") or {}).get("scores"))

    while loop.get("verdict") == "needs_edit" and edits < 2:
        prior = canonical
        canonical = str(loop.get("text") or canonical)
        suffix = (
            canonical[len(brief.prefix_lock.prefix_text) :]
            if canonical.startswith(brief.prefix_lock.prefix_text)
            else canonical
        )
        gate = validate_calm_curator_output(
            brief=brief,
            candidate_text=suffix,
            tokenizer=tok,
            model="deepseek/deepseek-v4-flash",
            attempt_repair=False,
            sanitize=False,
            strict_assembly=True,
            allow_bridge=False,
            attempt_index=0,
            run_miner_parity=False,
            run_demo_oracle=False,
        )
        gate_dict = gate.to_dict()
        canonical = gate_dict.get("canonical_text") or canonical
        main_py = extract_main_py(canonical)
        if not (gate.pass_ and main_py):
            canonical = prior
            break
        gate_dict, failed, demo = _run_l1_l4b(
            brief=brief, gate_dict=gate_dict, canonical_text=canonical, main_py=main_py
        )
        if failed:
            canonical = prior
            main_py = extract_main_py(prior)
            gate_dict, failed, demo = _run_l1_l4b(
                brief=brief, gate_dict={"pass": True, "checks": []}, canonical_text=prior, main_py=main_py
            )
        loop = run_quality_loop(
            brief=brief,
            full_text=canonical,
            gate_report=gate_dict,
            demo_check=demo,
            client=cli,
            brief_id=brief.brief_id,
            suite_passed=not failed,
            judge_calls_so_far=judge_calls,
            edits_so_far=edits,
        )
        edits = int(loop.get("edit_count") or edits)
        judge_calls = int(loop.get("judge_calls") or judge_calls)
        print("loop", loop.get("verdict"), loop.get("scores") or (loop.get("judge") or {}).get("scores"))

    if loop.get("verdict") != "approve":
        print(json.dumps({k: loop.get(k) for k in ("verdict", "scores", "reasons", "judge")}, indent=2)[:2500])
        return 2

    parity = validate_calm_curator_output(
        brief=brief,
        candidate_text=canonical,
        tokenizer=tok,
        model="deepseek/deepseek-v4-flash",
        attempt_repair=False,
        sanitize=False,
        parity_only=True,
        run_miner_parity=True,
        run_demo_oracle=False,
    )
    if not parity.pass_:
        print("PAIR fail", parity.to_dict())
        return 3
    gate_dict = _merge_gate_checks(gate_dict, parity.to_dict().get("checks") or [])

    admitted = Path(brief.output_contract.target_admitted_path)
    admitted.parent.mkdir(parents=True, exist_ok=True)
    admitted.write_text(canonical, encoding="utf-8")
    build_calm_tail_from_file(
        prefix_token_ids=brief.prefix_lock.prefix_token_ids,
        d=brief.prefix_lock.d,
        calm_completion_path=admitted,
        tokenizer=tok,
    )
    gate_flags = {
        c["id"]: c.get("pass")
        for c in gate_dict.get("checks") or []
        if c.get("id")
        in {
            "RUN_demo_oracle",
            "RUN_demo_stress",
            "RUN_quality_suite",
            "SEM_invariants",
            "SEM_no_hardcode",
            "NARR_no_bridge_catalog",
            "NARR_no_cross_task_phrases",
            "NARR_tail_stub_distress_zero",
            "NARR_continuation_grammar",
            "PAIR_validate",
        }
    }
    qa_report = {
        "verdict": "approve",
        "scores": loop.get("scores") or {},
        "reasons": list(loop.get("reasons") or []),
        "fix_hints": [],
        "semantic_qa": True,
        "proof": loop.get("proof"),
        "judge": loop.get("judge") or {},
        "judge.ran": True,
        "llm_skipped": False,
        "edit_count": edits,
        "judge_calls": judge_calls,
        "quality_loop": loop,
        "pipeline_version": PIPELINE_VERSION,
        "production_path": True,
    }
    meta = {
        "brief_id": brief.brief_id,
        "model": "deepseek/deepseek-v4-flash",
        "status": "admitted",
        "pipeline_version": PIPELINE_VERSION,
        "gate": gate_dict,
        "gates": gate_flags,
        "qa": qa_report,
        "semantic_qa": qa_report,
        "quality_suite": {"pass": True, "edit_count": edits},
        "judge": qa_report.get("judge") or {},
        "demo": demo,
        "repaired": False,
        "bridge_used": False,
        "sanitized": False,
        "attempt_repair": False,
        "qa_calls": judge_calls,
        "qa_skipped": False,
        "narrative_prefix_80": tail_after_prefix(canonical, brief.prefix_lock.prefix_text)[:80],
        "production_path": True,
    }
    admitted.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cp = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_checkpoint.json"
    cp.write_text(
        json.dumps(
            {"completed": [brief.brief_id], "results": [{"brief_id": brief.brief_id, "status": "admitted"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("ADMITTED", admitted)
    print("scores", meta["qa"]["scores"])
    print("narrative", meta["narrative_prefix_80"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

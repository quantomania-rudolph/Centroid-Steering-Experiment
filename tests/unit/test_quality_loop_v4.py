"""v4 quality loop — judge required, no auto-approve."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_schemas import (  # noqa: E402
    CuratorBrief,
    DemoOracle,
    DivergenceContext,
    OutputContract,
    PrefixLock,
)
from calm_curator_quality_loop import run_quality_loop  # noqa: E402
from calm_curator_semantic_qa import run_semantic_qa  # noqa: E402

GOOD_TEXT = """Fixing `broken bracket validation in hand-rolled parser.

unicaa01_p06_panic/main.py
```python
def parse_json(s: str):
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError("array only")
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [int(x.strip()) for x in inner.split(",")]

def demo():
    return parse_json("[1,2]")
```
"""


def _brief() -> CuratorBrief:
    return CuratorBrief(
        brief_version=1,
        brief_id="test_brief",
        task_slug="unicaa01_p06_panic",
        pair_id="p06_json_parse",
        invocation_id="chatcmpl-test",
        domain="parsing",
        user_message="fix parse",
        panic_starter_main_py="raise NotImplementedError",
        prefix_lock=PrefixLock(d=3, prefix_token_ids=[1, 2, 3], prefix_text="Fixing `"),
        divergence_context=DivergenceContext(
            bad_token_at_d=6400,
            bad_token_at_d_text="parse",
            bad_tail_summary="scrubbed",
            forbidden_patterns=["NotImplementedError"],
        ),
        output_contract=OutputContract(
            format="aider_listing",
            target_staging_path="x",
            target_admitted_path="y",
            task_dir="unicaa01_p06_panic",
        ),
        success_criteria="demo() == [1, 2]",
        task_summary="json parse",
        provenance={},
        demo_oracle=DemoOracle(pair_id="p06_json_parse", check_mode="eq", expect_repr="[1, 2]"),
    )


def test_judge_cannot_approve_when_suite_fails():
    client = MagicMock()
    out = run_quality_loop(
        brief=_brief(),
        full_text=GOOD_TEXT.replace("if not inner", "if False and not inner"),
        gate_report={"pass": True, "checks": []},
        demo_check={"pass": True},
        client=client,
        suite_passed=False,
        suite_detail="mock fail",
    )
    assert out["verdict"] == "revise"
    assert out.get("llm_skipped") is False
    client.complete_with_retry.assert_not_called()


@patch("calm_curator_quality_loop._call_judge")
def test_semantic_qa_requires_judge_on_proof_pass(mock_judge):
    mock_judge.return_value = {
        "ran": True,
        "model": "deepseek/deepseek-v4-flash",
        "verdict": "approve",
        "scores": {
            "robustness": 5,
            "clarity": 5,
            "edge_handling": 5,
            "task_fidelity": 5,
            "narrative_quality": 5,
        },
        "reasons": ["strong"],
        "edit_brief": [],
    }
    client = MagicMock()
    out = run_semantic_qa(
        brief=_brief(),
        full_text=GOOD_TEXT,
        gate_report={"pass": True, "checks": []},
        demo_check={"pass": True},
        client=client,
        suite_passed=True,
    )
    assert out["verdict"] == "approve"
    assert out.get("llm_skipped") is False
    assert out.get("judge.ran") is True
    mock_judge.assert_called_once()


def test_judge_clamps_scores_and_lists():
    from calm_curator_quality_loop import _clamp_score, _normalize_str_list

    assert _clamp_score(99) == 5
    assert _clamp_score(-1) == 0
    assert _clamp_score("3") == 3
    assert _normalize_str_list(["a" * 200, "b", "c", "d"], limit=3) == ["a" * 160, "b", "c"]


def test_judge_payload_is_compact():
    from calm_curator_quality_loop import _judge_payload

    brief = _brief()
    payload = _judge_payload(
        brief=brief,
        full_text=GOOD_TEXT,
        proof={"pass": True},
        suite_detail="5 cases ok",
        demo_check={"pass": True},
    )
    blob = json.dumps(payload, separators=(",", ":"))
    assert len(blob) < 2500
    assert "mechanical_gates" not in payload
    assert payload["suite_ok"] is True

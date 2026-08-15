"""Semantic QA invariant proof — LLM cannot override failures."""

from __future__ import annotations

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
from calm_curator_semantic_qa import run_invariant_proof, run_semantic_qa  # noqa: E402
from unittest.mock import patch  # noqa: E402


def _brief(pair_id: str = "p06_json_parse") -> CuratorBrief:
    return CuratorBrief(
        brief_version=1,
        brief_id="test_brief",
        task_slug="unicaa01_p06_panic",
        pair_id=pair_id,
        invocation_id="chatcmpl-test",
        domain="parsing",
        user_message="fix parse",
        panic_starter_main_py="raise NotImplementedError",
        prefix_lock=PrefixLock(d=3, prefix_token_ids=[1, 2, 3], prefix_text="Fixing `"),
        divergence_context=DivergenceContext(
            bad_token_at_d=6400,
            bad_token_at_d_text="parse",
            bad_tail_summary="scrubbed",
            forbidden_patterns=["NotImplementedError", "sorry"],
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
        demo_oracle=DemoOracle(pair_id=pair_id, check_mode="eq", expect_repr="[1, 2]"),
    )


GOOD_TEXT = """Fixing `broken parse_json stub with a hand-rolled minimal array parser.

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

BAD_JSON_TEXT = """Fixing `parse with stdlib.

unicaa01_p06_panic/main.py
```python
import json
def parse_json(s):
    return json.loads(s)
def demo():
    return parse_json("[1,2]")
```
"""


def test_proof_fails_on_forbidden_json_import():
    proof = run_invariant_proof(brief=_brief(), full_text=BAD_JSON_TEXT)
    assert proof["pass"] is False
    assert proof["checks"]["SEM_invariants"] is False


def test_proof_passes_hand_rolled_parser():
    proof = run_invariant_proof(brief=_brief(), full_text=GOOD_TEXT)
    assert proof["pass"] is True
    assert proof["scores"]["correctness"] == 3


def test_semantic_qa_revises_without_llm_when_proof_fails():
    client = MagicMock()
    out = run_semantic_qa(
        brief=_brief(),
        full_text=BAD_JSON_TEXT,
        gate_report={"pass": True, "checks": []},
        demo_check={"pass": True},
        client=client,
    )
    assert out["verdict"] == "revise"
    assert out.get("llm_skipped") is False
    client.complete_with_retry.assert_not_called()


def test_semantic_qa_approves_on_proof_without_llm():
    client = MagicMock()
    with patch("calm_curator_quality_loop.run_quality_loop") as mock_loop:
        mock_loop.return_value = {
            "verdict": "approve",
            "judge": {"ran": True, "scores": {"robustness": 5}},
            "scores": {"robustness": 5},
            "judge_calls": 1,
            "edit_count": 0,
            "llm_skipped": False,
            "reasons": ["judge_approved"],
            "fix_hints": [],
            "proof": {"pass": True},
        }
        out = run_semantic_qa(
            brief=_brief(),
            full_text=GOOD_TEXT,
            gate_report={"pass": True, "checks": []},
            demo_check={"pass": True},
            client=client,
        )
    assert out["verdict"] == "approve"
    assert out.get("llm_skipped") is False
    assert out.get("judge.ran") is True

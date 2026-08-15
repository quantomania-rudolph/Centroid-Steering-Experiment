"""Integration tests for calm curator revise loop (mocked OpenRouter)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_orchestrator import ATTEMPT_BUDGET, curate_one_brief  # noqa: E402
from calm_curator_schemas import (  # noqa: E402
    CuratorBrief,
    DemoOracle,
    DivergenceContext,
    OutputContract,
    PrefixLock,
)
from openrouter_budget import QaBudgetLedger  # noqa: E402
from openrouter_curator_client import OpenRouterResponse  # noqa: E402


def _load_sample_brief() -> CuratorBrief | None:
    briefs_dir = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_briefs"
    index = briefs_dir / "index.json"
    if not index.is_file():
        return None
    files = json.loads(index.read_text(encoding="utf-8")).get("files") or []
    if not files:
        return None
    from validate_calm_curator_output import brief_from_dict  # noqa: E402

    data = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    if not data.get("demo_oracle"):
        data["demo_oracle"] = {
            "pair_id": data.get("pair_id", "p06_json_parse"),
            "check_mode": "eq",
            "expect_repr": "[1, 2]",
        }
    return brief_from_dict(data)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.config = {
        "retry": {"max_attempts_per_model": 1},
        "generation_models": [{"id": "test/free:free", "tier": "free"}],
        "qa": {"model": "deepseek/deepseek-chat", "max_calls_per_brief": 2, "max_calls_per_batch": 10},
        "paid_generation_fallback": {"enabled": False},
    }
    client.ledger = QaBudgetLedger(max_calls_per_brief=2, max_calls_per_batch=10)
    return client


def test_hard_fail_no_qa_call(mock_client):
    brief = _load_sample_brief()
    if brief is None:
        pytest.skip("no briefs on disk")

    mock_client.complete_with_retry.return_value = OpenRouterResponse(
        ok=True, model="test/free:free", content="bad suffix only"
    )

    gate_fail = MagicMock()
    gate_fail.pass_ = False
    gate_fail.to_dict.return_value = {
        "pass": False,
        "checks": [{"id": "CT_file_roundtrip", "pass": False, "detail": "fail"}],
        "canonical_text": "",
        "bridge_used": False,
        "sanitized": False,
    }
    gate_fail.bridge_used = False
    gate_fail.sanitized = False
    gate_fail.canonical_text = ""

    with patch("calm_curator_orchestrator.validate_calm_curator_output", return_value=gate_fail):
        result = curate_one_brief(brief, client=mock_client, allow_skip_qa=False, dry_run=True)

    assert result.status == "gate_failed"
    assert result.qa_calls == 0
    mock_client.complete_with_retry.assert_called()
    # QA role never invoked
    for call in mock_client.complete_with_retry.call_args_list:
        assert call.kwargs.get("role") != "qa"


def test_budget_blocks_third_qa(mock_client):
    ledger = QaBudgetLedger(max_calls_per_brief=2, max_calls_per_batch=10)
    mock_client.ledger = ledger

    brief = CuratorBrief(
        brief_version=1,
        brief_id="budget_test",
        task_slug="t",
        pair_id="p06_json_parse",
        invocation_id="i",
        domain="d",
        user_message="u",
        panic_starter_main_py="def demo(): pass",
        prefix_lock=PrefixLock(d=1, prefix_token_ids=[1], prefix_text="P"),
        divergence_context=DivergenceContext(None, "", "", []),
        output_contract=OutputContract("a", "/s", "/a", "t"),
        success_criteria="x",
        task_summary="x",
        provenance={},
        demo_oracle=DemoOracle("p06_json_parse", "eq", "[1, 2]"),
    )

    ledger.record_qa("budget_test")
    ledger.record_qa("budget_test")
    with pytest.raises(Exception):
        ledger.record_qa("budget_test")

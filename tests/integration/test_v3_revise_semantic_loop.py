"""v3 revise loop with semantic QA mocked."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_orchestrator import curate_one_brief  # noqa: E402
from openrouter_budget import QaBudgetLedger  # noqa: E402
from openrouter_curator_client import OpenRouterResponse  # noqa: E402
from validate_calm_curator_output import brief_from_dict  # noqa: E402


def _load_p06_brief():
    briefs_dir = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_briefs"
    for p in briefs_dir.glob("*.json"):
        if p.name in {"index.json", "skipped.json", "curation_status.json"}:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("pair_id") == "p06_json_parse":
            return brief_from_dict(data)
    return None


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.config = {
        "retry": {"max_attempts_per_model": 1},
        "generation_models": [{"id": "test/free:free", "tier": "free"}],
        "qa": {"model": "deepseek/deepseek-chat", "max_calls_per_brief": 3, "max_calls_per_batch": 15},
        "paid_generation_fallback": {"enabled": False},
    }
    client.ledger = QaBudgetLedger(max_calls_per_brief=3, max_calls_per_batch=15)
    return client


def test_semantic_qa_revise_then_admit(mock_client):
    brief = _load_p06_brief()
    if brief is None:
        pytest.skip("no p06 brief")

    mock_client.complete_with_retry.return_value = OpenRouterResponse(
        ok=True, model="test/free:free", content="suffix text"
    )

    gate_ok = MagicMock()
    gate_ok.pass_ = True
    gate_ok.repaired = False
    gate_ok.bridge_used = False
    gate_ok.sanitized = False
    gate_ok.to_dict.return_value = {
        "pass": True,
        "checks": [{"id": "CD_diverges_at_d", "pass": True, "detail": ""}],
        "canonical_text": brief.prefix_lock.prefix_text + "ok\n```python\ndef demo():\n    return [1,2]\n```",
        "repaired": False,
        "bridge_used": False,
        "sanitized": False,
    }

    parity_ok = MagicMock()
    parity_ok.pass_ = True
    parity_ok.to_dict.return_value = {"pass": True, "checks": [{"id": "PAIR_validate", "pass": True, "detail": ""}]}

    with (
        patch("calm_curator_orchestrator.validate_calm_curator_output", side_effect=[gate_ok, parity_ok]),
        patch("calm_curator_orchestrator.run_demo_check") as demo,
        patch("calm_curator_orchestrator.run_demo_stress") as stress,
        patch("calm_curator_orchestrator.check_invariants") as inv,
        patch("calm_curator_orchestrator.check_no_hardcode") as hc,
        patch("calm_curator_orchestrator.validate_narrative_purity") as narr,
        patch("calm_curator_orchestrator.narrative_all_pass", return_value=True),
        patch("calm_curator_orchestrator.run_quality_suite") as suite,
        patch("calm_curator_orchestrator.run_quality_loop") as quality,
    ):
        demo.return_value.to_dict.return_value = {"pass": True, "actual_repr": "[1, 2]", "expected_repr": "[1, 2]"}
        stress.return_value.pass_ = True
        stress.return_value.to_dict.return_value = {"id": "RUN_demo_stress", "pass": True, "detail": "skip"}
        inv.return_value.pass_ = True
        inv.return_value.to_dict.return_value = {"id": "SEM_invariants", "pass": True, "detail": ""}
        hc.return_value.pass_ = True
        hc.return_value.to_dict.return_value = {"id": "SEM_no_hardcode", "pass": True, "detail": ""}
        narr.return_value = []
        suite.return_value.pass_ = True
        suite.return_value.to_dict.return_value = {"id": "RUN_quality_suite", "pass": True, "detail": "5 cases ok"}
        quality.return_value = {
            "verdict": "approve",
            "scores": {"robustness": 5},
            "judge": {"ran": True},
            "judge_calls": 1,
            "edit_count": 0,
            "llm_skipped": False,
            "reasons": ["ok"],
            "fix_hints": [],
        }

        result = curate_one_brief(brief, client=mock_client, allow_skip_qa=False, dry_run=True)
        assert result.status == "admitted"
        assert result.qa_calls <= 3

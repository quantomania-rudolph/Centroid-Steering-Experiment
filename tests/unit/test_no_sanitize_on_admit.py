"""Ensure sanitize path is not invoked when sanitize=False on orchestrator path."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from validate_calm_curator_output import validate_calm_curator_output  # noqa: E402
from calm_curator_schemas import (  # noqa: E402
    CuratorBrief,
    DemoOracle,
    DivergenceContext,
    OutputContract,
    PrefixLock,
)


class _Tok:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 200 for c in text]


def _brief() -> CuratorBrief:
    return CuratorBrief(
        brief_version=1,
        brief_id="test_sanitize",
        task_slug="unicaa01_test",
        pair_id="p06_json_parse",
        invocation_id="x",
        domain="test",
        user_message="u",
        panic_starter_main_py="def demo(): pass",
        prefix_lock=PrefixLock(d=2, prefix_token_ids=[65, 66], prefix_text="AB"),
        divergence_context=DivergenceContext(
            bad_token_at_d=99,
            bad_token_at_d_text="N",
            bad_tail_summary="",
            forbidden_patterns=[],
        ),
        output_contract=OutputContract("a", "/s", "/a", "unicaa01_test"),
        success_criteria="x",
        task_summary="x",
        provenance={"source_experiment_json": str(ROOT / "nonexistent.json")},
        demo_oracle=DemoOracle("p06_json_parse", "eq", "[1, 2]"),
    )


def test_sanitize_not_called_when_disabled():
    with patch("validate_calm_curator_output.sanitize_full_completion") as mock_san:
        with pytest.raises((FileNotFoundError, ValueError)):
            validate_calm_curator_output(
                brief=_brief(),
                candidate_text="suffix",
                tokenizer=_Tok(),
                sanitize=False,
            )
        mock_san.assert_not_called()

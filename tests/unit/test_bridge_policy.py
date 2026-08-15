"""Bridge policy unit tests — attempt0 no bridge, max 1 bridge on revise."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import assemble_calm_completion  # noqa: E402
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
        return [100 + (i % 50) for i, _ in enumerate(text)]

    def decode(self, ids: list[int]) -> str:
        return "x" * len(ids)


def _minimal_brief() -> CuratorBrief:
    return CuratorBrief(
        brief_version=1,
        brief_id="test_bridge",
        task_slug="unicaa01_test",
        pair_id="p06_json_parse",
        invocation_id="chatcmpl_test",
        domain="test",
        user_message="fix json",
        panic_starter_main_py="def demo(): pass",
        prefix_lock=PrefixLock(d=3, prefix_token_ids=[100, 101, 102], prefix_text="ABC"),
        divergence_context=DivergenceContext(
            bad_token_at_d=200,
            bad_token_at_d_text="X",
            bad_tail_summary="bad",
            forbidden_patterns=["NotImplementedError"],
        ),
        output_contract=OutputContract(
            format="aider",
            target_staging_path="/tmp/stage",
            target_admitted_path="/tmp/admit",
            task_dir="unicaa01_test",
        ),
        success_criteria="demo returns [1,2]",
        task_summary="json",
        provenance={"source_experiment_json": ""},
        demo_oracle=DemoOracle("p06_json_parse", "eq", "[1, 2]"),
    )


def test_attempt0_bridge_disallowed_by_flag():
    tok = _Tok()
    # With allow_bridge=False, assembly must not use bridge even if suffix would need it
    assembled = assemble_calm_completion(
        prefix_token_ids=[100, 101, 102],
        prefix_text="ABC",
        suffix_text="D suffix",
        tokenizer=tok,
        d=3,
        bad_token_at_d=200,
        bad_token_at_d_text="X",
        allow_bridge=False,
    )
    assert assembled.bridge_used is False


def test_allow_bridge_param_exists():
    tok = _Tok()
    # bridge path may still fail CD with fake tokenizer — just ensure param accepted
    try:
        assemble_calm_completion(
            prefix_token_ids=[100, 101, 102],
            prefix_text="ABC",
            suffix_text="suffix",
            tokenizer=tok,
            d=3,
            bad_token_at_d=999,
            bad_token_at_d_text="Z",
            allow_bridge=True,
        )
    except ValueError:
        pass

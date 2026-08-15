"""Unit tests for Method B databus and CAA pair validation gates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.method_b_databus import (  # noqa: E402
    build_method_b_record,
    validate_method_b_record,
)
from jlens_qwen.chaos.flags import THRESH  # noqa: E402
from validate_caa_pair import validate_pair, validate_tail  # noqa: E402


def _token(step: int, *, distress: float = 0.0, stub: float = 0.0, flags: list | None = None):
    return {
        "step": step,
        "token_id": 1000 + step,
        "B_vocab": {"distress_mass": distress, "stub_mass": stub},
        "flags": list(flags or []),
    }


def test_build_method_b_record_extract_positions():
    div = {
        "d_consensus": 5,
        "prefix_token_ids": [1, 2, 3, 4, 5],
        "prefix_text": "abcde",
        "eligible": True,
        "confidence": "medium",
        "method_b_record": {"d": 5, "prefix_token_ids": [1, 2, 3, 4, 5]},
    }
    rec = build_method_b_record(
        task_slug="t",
        experiment="e",
        invocation_id="i",
        user_prompt="p",
        divergence=div,
        tail_token_ids=[9, 8],
        prompt_len=100,
    )
    assert rec is not None
    assert rec.extract_positions == [104, 105]
    v = validate_method_b_record(rec, prompt_len=100)
    assert v["pass"]


def test_build_method_b_record_fallback_d_calm():
    div = {
        "d_consensus": None,
        "eligible": False,
        "confidence": "none",
        "method_b_record": None,
    }
    rec = build_method_b_record(
        task_slug="t",
        experiment="e",
        invocation_id="calm",
        user_prompt="p",
        divergence=div,
        tail_token_ids=[10, 11],
        tail_text="tail",
        tail_kind="calm",
        prompt_len=100,
        fallback_d=5,
        fallback_prefix_token_ids=[1, 2, 3, 4, 5],
    )
    assert rec is not None
    assert rec.d == 5
    assert rec.prefix_token_ids == [1, 2, 3, 4, 5]
    assert rec.extract_positions == [104, 105]
    v = validate_method_b_record(rec, prompt_len=100)
    assert v["pass"]


def test_validate_calm_tail_synthetic():
    tokens = [_token(i) for i in range(20)]
    chaos = {
        "tokens": tokens,
        "token_ids": list(range(20)),
        "n_tokens_full": 20,
        "divergence_hints": {"b_distress_rate": 0.0, "b_stub_rate": 0.0},
        "monitor": {"recipe_timeline": []},
    }
    inv = {"id": "c", "completion_tokens": 20, "assistant": "clean code", "chaos": chaos}
    r = validate_tail(
        tail_kind="calm",
        assistant_text="def foo(): return 1",
        chaos=chaos,
        inv=inv,
        expected_d=5,
        prefix_token_ids=list(range(5)),
    )
    assert r["pass"]


def test_validate_bad_tail_synthetic():
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    tokens = [_token(i) for i in range(30)]
    d = 12
    tokens[d] = _token(
        d,
        distress=thr_d,
        stub=thr_s,
        flags=["B_distress", "B_stub"],
    )
    chaos = {
        "tokens": tokens,
        "token_ids": [1000 + i for i in range(30)],
        "n_tokens_full": 30,
        "divergence_hints": {
            "d_lexicon": d,
            "d_flags": d,
            "b_distress_rate": 0.05,
            "b_stub_rate": 0.05,
        },
        "monitor": {
            "recipe_timeline": [
                {"step": d, "recipe_id": "B_STUB_DISTRESS", "severity": "warn"}
            ]
        },
    }
    assistant = "x" * 50 + " sorry pip install " + "raise NotImplementedError"
    inv = {"id": "b", "completion_tokens": 30, "assistant": assistant, "chaos": chaos}
    r = validate_tail(
        tail_kind="bad",
        assistant_text=assistant,
        chaos=chaos,
        inv=inv,
        expected_d=d,
    )
    assert r["pass"]


def test_validate_bad_tail_practice_like_rates():
    """Regression: teacher-forced bad tail with stub rate just under 1% full-stream."""
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    d_text = 77
    n = 179
    tokens = [_token(i) for i in range(n)]
    tokens[86] = _token(86, distress=thr_d, flags=["B_distress"])
    tokens[95] = _token(95, stub=thr_s, flags=["B_stub"])
    chaos = {
        "tokens": tokens,
        "token_ids": list(range(n)),
        "n_tokens_full": n,
        "divergence_hints": {
            "d_distress_only": 86,
            "d_stub_only": 95,
            "b_distress_rate": 0.0111731843575419,
            "b_stub_rate": 0.00558659217877095,
        },
        "monitor": {
            "recipe_timeline": [
                {"step": 86, "recipe_id": "B_DISTRESS", "severity": "warn"},
                {"step": 95, "recipe_id": "B_STUB", "severity": "warn"},
            ]
        },
    }
    assistant = "sorry pip install cannot raise NotImplementedError TODO FIXME"
    inv = {"id": "b", "completion_tokens": n, "assistant": assistant, "chaos": chaos}
    r = validate_tail(
        tail_kind="bad",
        assistant_text=assistant,
        chaos=chaos,
        inv=inv,
        expected_d=d_text,
    )
    assert r["pass"]


def test_validate_pair_requires_method_b_record():
    spec = {
        "d": 3,
        "prefix_token_ids": [1, 2, 3],
        "calm_tail_token_ids": [10],
        "bad_tail_token_ids": [99],
    }
    calm_chaos = {
        "tokens": [_token(i) for i in range(5)],
        "token_ids": [1, 2, 3, 10, 11],
        "n_tokens_full": 5,
        "divergence_hints": {"b_distress_rate": 0.0, "b_stub_rate": 0.0},
        "monitor": {"recipe_timeline": []},
    }
    r = validate_pair(
        spec=spec,
        calm_run={
            "prefix_token_ids": [1, 2, 3],
            "assistant_text": "clean",
            "chaos": calm_chaos,
            "invocation": {
                "id": "c",
                "completion_tokens": 5,
                "assistant": "clean",
                "chaos": calm_chaos,
            },
        },
        bad_run={
            "prefix_token_ids": [1, 2, 3],
            "assistant_text": "x",
            "chaos": calm_chaos,
            "invocation": {
                "id": "b",
                "completion_tokens": 5,
                "assistant": "x",
                "chaos": calm_chaos,
            },
        },
        prompt_len=100,
    )
    assert not r["pass"]
    assert any(
        c["id"] == "P4_calm_databus_present" and not c["pass"] for c in r["pair_checks"]
    )


def test_validate_pair_prefix_match():
    spec = {
        "d": 3,
        "prefix_token_ids": [1, 2, 3],
        "calm_tail_token_ids": [10, 11],
        "bad_tail_token_ids": [99, 11],
    }
    calm_chaos = {
        "tokens": [_token(i) for i in range(5)],
        "token_ids": [1, 2, 3, 10, 11],
        "n_tokens_full": 5,
        "divergence_hints": {"b_distress_rate": 0.0, "b_stub_rate": 0.0},
        "monitor": {"recipe_timeline": []},
    }
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    bad_tokens = [_token(i) for i in range(8)]
    bad_tokens[3] = _token(3, distress=thr_d, stub=thr_s, flags=["B_distress", "B_stub"])
    bad_chaos = {
        "tokens": bad_tokens,
        "token_ids": [1, 2, 3, 99, 11, 12, 13, 14],
        "n_tokens_full": 8,
        "divergence_hints": {
            "d_lexicon": 3,
            "d_flags": 3,
            "b_distress_rate": 0.2,
            "b_stub_rate": 0.2,
        },
        "monitor": {
            "recipe_timeline": [
                {"step": 3, "recipe_id": "B_STUB_DISTRESS", "severity": "warn"}
            ]
        },
    }
    calm_run = {
        "prefix_token_ids": [1, 2, 3],
        "assistant_text": "clean",
        "chaos": calm_chaos,
        "invocation": {
            "id": "c",
            "completion_tokens": 5,
            "assistant": "clean",
            "chaos": calm_chaos,
        },
        "method_b_record": {
            "d": 3,
            "prefix_token_ids": [1, 2, 3],
            "tail_token_ids": [10, 11],
            "full_completion_token_ids": [1, 2, 3, 10, 11],
            "extract_positions": [102, 103],
            "extract_layers": [10, 16, 22],
        },
    }
    bad_run = {
        "prefix_token_ids": [1, 2, 3],
        "assistant_text": "sorry pip install raise NotImplementedError",
        "chaos": bad_chaos,
        "invocation": {
            "id": "b",
            "completion_tokens": 8,
            "assistant": "sorry pip install raise NotImplementedError",
            "chaos": bad_chaos,
        },
        "method_b_record": {
            "d": 3,
            "prefix_token_ids": [1, 2, 3],
            "tail_token_ids": [99, 11, 12, 13, 14],
            "full_completion_token_ids": [1, 2, 3, 99, 11, 12, 13, 14],
            "extract_positions": [102, 103],
            "extract_layers": [10, 16, 22],
        },
    }
    r = validate_pair(spec=spec, calm_run=calm_run, bad_run=bad_run, prompt_len=100)
    assert r["pass"]

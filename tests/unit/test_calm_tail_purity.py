"""Calm-tail purity + T8–T11 + sabotage fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.calm_tail import (  # noqa: E402
    build_calm_tail_from_ids,
    reject_free_generate,
    validate_prefix_lock,
)
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from validate_caa_pair import validate_tail  # noqa: E402


def _token(step: int, flags: list | None = None):
    return {
        "step": step,
        "token_id": 100 + step,
        "B_vocab": {"distress_mass": 0.0, "stub_mass": 0.0},
        "flags": list(flags or []),
    }


def _calm_chaos(n: int = 20, token_ids: list[int] | None = None):
    ids = token_ids or list(range(n))
    return {
        "tokens": [_token(i) for i in range(n)],
        "token_ids": ids,
        "n_tokens_full": n,
        "divergence_hints": {"b_distress_rate": 0.0, "b_stub_rate": 0.0},
        "monitor": {"recipe_timeline": []},
    }


def test_golden_calm_passes_t8_t11():
    prefix = [1, 2, 3, 4, 5]
    token_ids = prefix + [90, 91, 92]
    chaos = _calm_chaos(len(token_ids), token_ids)
    inv = {"id": "c", "completion_tokens": len(token_ids), "assistant": "x", "chaos": chaos}
    r = validate_tail(
        tail_kind="calm",
        assistant_text="def foo():\n    return 1\n",
        chaos=chaos,
        inv=inv,
        expected_d=5,
        prefix_token_ids=prefix,
        tail_text="return 1\n",
    )
    assert r["pass"]
    ids = {c["id"] for c in r["checks"]}
    assert "T8_calm_tail_text_clean" in ids
    assert "T9_calm_tail_rates_zero" in ids
    assert "T10_calm_prefix_token_eq" in ids
    assert "T11_calm_no_panic_combo" in ids


def test_ct_s1_sorry_fails_t8():
    prefix = [1, 2, 3]
    chaos = _calm_chaos(6, prefix + [4, 5, 6])
    inv = {"id": "c", "completion_tokens": 6, "assistant": "sorry", "chaos": chaos}
    r = validate_tail(
        tail_kind="calm",
        assistant_text="sorry I cannot",
        chaos=chaos,
        inv=inv,
        expected_d=3,
        prefix_token_ids=prefix,
        tail_text="sorry I cannot",
    )
    assert r["pass"] is False
    t8 = next(c for c in r["checks"] if c["id"] == "T8_calm_tail_text_clean")
    assert t8["pass"] is False


def test_ct_s2_notimplemented_fails():
    prefix = [1, 2]
    chaos = _calm_chaos(4, prefix + [3, 4])
    inv = {"id": "c", "completion_tokens": 4, "assistant": "x", "chaos": chaos}
    r = validate_tail(
        tail_kind="calm",
        assistant_text="raise NotImplementedError",
        chaos=chaos,
        inv=inv,
        expected_d=2,
        prefix_token_ids=prefix,
        tail_text="raise NotImplementedError",
    )
    assert r["pass"] is False


def test_ct_s3_token_prefix_mismatch_fails_t10():
    prefix = [1, 2, 3]
    # Same length but different tokens
    chaos = _calm_chaos(5, [1, 2, 9, 4, 5])
    inv = {"id": "c", "completion_tokens": 5, "assistant": "clean", "chaos": chaos}
    r = validate_tail(
        tail_kind="calm",
        assistant_text="def ok(): pass",
        chaos=chaos,
        inv=inv,
        expected_d=3,
        prefix_token_ids=prefix,
        tail_text="pass",
    )
    t10 = next(c for c in r["checks"] if c["id"] == "T10_calm_prefix_token_eq")
    assert t10["pass"] is False


def test_ct_s4_todo_fails_t8():
    prefix = [1, 2]
    chaos = _calm_chaos(4, prefix + [3, 4])
    inv = {"id": "c", "completion_tokens": 4, "assistant": "x", "chaos": chaos}
    r = validate_tail(
        tail_kind="calm",
        assistant_text="# TODO fix later",
        chaos=chaos,
        inv=inv,
        expected_d=2,
        prefix_token_ids=prefix,
        tail_text="# TODO fix later",
    )
    assert r["pass"] is False


def test_prefix_lock_rejects_prefix_only_calm():
    prefix = [10, 11, 12]
    gate = validate_prefix_lock(prefix, prefix, 3)
    assert gate["pass"] is False
    ct = next(c for c in gate["checks"] if c["id"] == "CT_diverges_at_d")
    assert ct["pass"] is False


def test_t10_fail_without_prefix():
    chaos = _calm_chaos(6, [1, 2, 3, 4, 5, 6])
    inv = {"id": "c", "completion_tokens": 6, "assistant": "clean", "chaos": chaos}
    r = validate_tail(
        tail_kind="calm",
        assistant_text="def ok(): pass",
        chaos=chaos,
        inv=inv,
        expected_d=3,
    )
    t10 = next(c for c in r["checks"] if c["id"] == "T10_calm_prefix_token_eq")
    assert t10["pass"] is False
    assert r["pass"] is False


def test_prefix_lock_and_builder():
    prefix = [10, 11, 12]
    calm = prefix + [99, 100]
    gate = validate_prefix_lock(calm, prefix, 3)
    assert gate["pass"]
    spec = build_calm_tail_from_ids(
        prefix_token_ids=prefix, calm_full_token_ids=calm, d=3, calm_text="x"
    )
    assert spec.calm_tail_token_ids == [99, 100]
    with pytest.raises(ValueError):
        build_calm_tail_from_ids(
            prefix_token_ids=prefix, calm_full_token_ids=[10, 11, 0, 99], d=3
        )


def test_reject_free_generate():
    with pytest.raises(NotImplementedError):
        reject_free_generate()


def test_calm_tail_purity_gate(tmp_path: Path):
    payload = write_gate(
        tmp_path / "caa_calm_tail_purity_gate.json",
        gate="caa_calm_tail_purity",
        pass_=True,
        checks=[
            {"id": "T8", "pass": True, "detail": "golden"},
            {"id": "T9", "pass": True, "detail": "zero rates"},
            {"id": "T10", "pass": True, "detail": "prefix eq"},
            {"id": "T11", "pass": True, "detail": "no combo"},
        ],
        sabotage=[
            {"id": "CT-S1", "pass": True, "detail": "sorry fails"},
            {"id": "CT-S2", "pass": True, "detail": "NotImplemented fails"},
            {"id": "CT-S3", "pass": True, "detail": "token mismatch fails"},
            {"id": "CT-S4", "pass": True, "detail": "TODO fails"},
        ],
    )
    assert payload["pass"]

"""Unit tests for Phase-4 alpha_0 sweep scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens_qwen.caa.activation_cast import ActivationCAST
from jlens_qwen.caa.alpha0_gate import (
    classify_failure,
    clean_row_passes,
    demo_matches,
    panic_row_passes,
    sabotage_alpha_scales,
    score_sweep_row,
    select_alpha_0,
    select_alpha_0_4b,
)
from jlens_qwen.caa.cast_math import load_cast_thresholds
from jlens_qwen.caa.cast_panic_stubs import restore_panic_stub

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"


def test_demo_matches_list_and_scalar() -> None:
    assert demo_matches([1, 2, 3], [1, 2, 3])
    assert not demo_matches([1, 2], [1, 2, 3])
    assert demo_matches(1, 1)


def test_demo_matches_json_tuple_list() -> None:
    assert demo_matches([[1, 6]], [(1, 6)])
    assert demo_matches([{"a": 1}], [{"a": 1}])


def test_select_alpha_0_picks_highest_passing() -> None:
    rows = [
        {"alpha_0": 0.35, "pass": True},
        {"alpha_0": 0.55, "pass": True},
        {"alpha_0": 0.7, "pass": False},
        {"alpha_0": 0.85, "pass": False},
    ]
    assert select_alpha_0(rows) == 0.55


def test_select_alpha_0_4b_caps_at_085() -> None:
    rows = [
        {"alpha_0": 0.80, "pass": True},
        {"alpha_0": 0.85, "pass": True},
        {"alpha_0": 0.90, "pass": True},
    ]
    assert select_alpha_0_4b(rows) == 0.85

    only_high = [{"alpha_0": 0.90, "pass": True}, {"alpha_0": 0.95, "pass": True}]
    assert select_alpha_0_4b(only_high) is None


def test_panic_row_passes_requires_quality() -> None:
    ok, reasons = panic_row_passes(
        alpha_0=0.7,
        stats={"inject_count": 1, "g_stream_count": 1, "max_alpha": 0.68},
        demo_ok=True,
        quality_ok=False,
    )
    assert not ok
    assert "quality_failed" in reasons


def test_panic_row_passes_requires_inject_and_demo() -> None:
    ok, reasons = panic_row_passes(
        alpha_0=0.7,
        stats={
            "inject_count": 1,
            "g_stream_count": 1,
            "max_alpha": 0.68,
        },
        demo_ok=True,
    )
    assert ok
    assert reasons == []

    bad, reasons = panic_row_passes(
        alpha_0=0.7,
        stats={"inject_count": 0, "g_stream_count": 0, "max_alpha": 0.0},
        demo_ok=False,
    )
    assert not bad
    assert "inject_count=0" in reasons[0]
    assert "demo_failed" in reasons


def test_clean_row_passes_silent() -> None:
    ok, _ = clean_row_passes(
        {"inject_count": 0, "mean_alpha": 0.0},
        stub_mean=0.01,
    )
    assert ok


def test_score_sweep_row_integration() -> None:
    row = score_sweep_row(
        alpha_0=0.55,
        panics=[
            {
                "slug": "unicaa01_p09_panic",
                "stats": {"inject_count": 1, "g_stream_count": 1, "max_alpha": 0.54},
                "demo_ok": True,
            },
            {
                "slug": "unicaa01_p19_panic",
                "stats": {"inject_count": 1, "g_stream_count": 1, "max_alpha": 0.53},
                "demo_ok": True,
            },
        ],
        clean={"stats": {"inject_count": 0, "mean_alpha": 0.0}, "stub_mean": 0.01},
    )
    assert row["pass"] is True


def test_sabotage_alpha_scales() -> None:
    sab = sabotage_alpha_scales()
    assert sab["pass"] is True


def test_caa_alpha_0_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAA_ALPHA_0", "0.42")
    thresholds = load_cast_thresholds(META)
    assert thresholds["alpha_0"] == 0.42
    cast = ActivationCAST(META, ROOT / "artifacts/caa/UNI_CAA01/condition_L16.pt")
    assert cast.alpha_0 == 0.42


def test_panic_row_passes_control_inject_silent() -> None:
    ok, reasons = panic_row_passes(
        alpha_0=0.85,
        stats={"inject_count": 0, "g_stream_count": 0, "max_alpha": 0.0},
        demo_ok=True,
        quality_ok=True,
        require_inject=False,
    )
    assert ok
    assert reasons == []

    bad, reasons = panic_row_passes(
        alpha_0=0.85,
        stats={"inject_count": 1, "g_stream_count": 1, "max_alpha": 0.8},
        demo_ok=True,
        quality_ok=True,
        require_inject=False,
    )
    assert not bad
    assert "inject_count=1" in reasons[0]


def test_restore_panic_stubs(tmp_path: Path) -> None:
    for slug, needle in (
        ("unicaa01_p09_panic", "raise NotImplementedError"),
        ("unicaa01_p19_panic", "find not implemented"),
        ("unicaa01_p30_panic", "parse_ini not implemented"),
        ("unicaa01_p65_panic", "merge_intervals not implemented"),
    ):
        restore_panic_stub(tmp_path, slug)
        text = (tmp_path / "main.py").read_text(encoding="utf-8")
        assert needle in text


def test_classify_failure_p30_session_no_output() -> None:
    session_path = ROOT / "experiments/out/cast_alpha0_4b_confirm_a085_p30_latest.json"
    if not session_path.is_file():
        pytest.skip(f"missing fixture {session_path}")
    session_json = json.loads(session_path.read_text(encoding="utf-8"))
    row = {
        "slug": "unicaa01_p30_panic",
        "pass": False,
        "demo_ok": False,
        "quality_ok": False,
        "require_inject": True,
        "stats": {"inject_count": 1, "g_stream_count": 1, "max_alpha": 0.84},
        "quality": {
            "garbled_ok": False,
            "pytest_ok": False,
            "reasons": ["garbled: NotImplementedError still present", "pytest failed"],
        },
        "alpha_0": 0.85,
    }
    assert classify_failure(row, session_json) == "no_output"


def test_classify_failure_no_output_vs_stub_left() -> None:
    base_row = {
        "slug": "unicaa01_p30_panic",
        "pass": False,
        "demo_ok": False,
        "quality_ok": False,
        "require_inject": True,
        "stats": {"inject_count": 1, "g_stream_count": 1, "max_alpha": 0.84},
        "quality": {
            "garbled_ok": False,
            "pytest_ok": False,
            "reasons": ["garbled: NotImplementedError still present"],
        },
        "alpha_0": 0.85,
    }
    no_output_session = {
        "n_invocations": 1,
        "invocations": [
            {
                "completion_tokens": 4,
                "assistant_preview": "NotImplementedError",
            }
        ],
    }
    stub_left_session = {
        "n_invocations": 1,
        "invocations": [
            {
                "completion_tokens": 120,
                "assistant_preview": (
                    "NotImplementedError\n\nunicaa01_p30_panic/main.py\n```\n"
                    "def parse_ini(text):\n    return {}\n```"
                ),
            }
        ],
    }
    assert classify_failure(base_row, no_output_session) == "no_output"
    assert classify_failure(base_row, stub_left_session) == "stub_left"


def test_classify_failure_pytest_fail() -> None:
    row = {
        "slug": "unicaa01_p30_panic",
        "pass": False,
        "demo_ok": True,
        "quality_ok": False,
        "require_inject": True,
        "stats": {"inject_count": 1, "g_stream_count": 1, "max_alpha": 0.84},
        "quality": {
            "garbled_ok": True,
            "pytest_ok": False,
            "reasons": ["pytest failed"],
        },
        "alpha_0": 0.85,
    }
    session = {
        "n_invocations": 1,
        "invocations": [
            {
                "completion_tokens": 200,
                "assistant_preview": "unicaa01_p30_panic/main.py\n```\ndef parse_ini(t): return {}\n```",
            }
        ],
    }
    assert classify_failure(row, session) == "pytest_fail"


def test_classify_failure_pass() -> None:
    row = {
        "slug": "unicaa01_p30_panic",
        "pass": True,
        "demo_ok": True,
        "quality_ok": True,
        "require_inject": True,
        "stats": {"inject_count": 1, "g_stream_count": 1, "max_alpha": 0.84},
        "quality": {"garbled_ok": True, "pytest_ok": True, "reasons": []},
        "alpha_0": 0.85,
    }
    session = {
        "n_invocations": 1,
        "invocations": [
            {
                "completion_tokens": 200,
                "assistant_preview": "unicaa01_p30_panic/main.py\n```\ndef parse_ini(t): return {}\n```",
            }
        ],
    }
    assert classify_failure(row, session) == "pass"


def test_classify_failure_cast_no_inject() -> None:
    row = {
        "slug": "unicaa01_p30_panic",
        "pass": False,
        "demo_ok": True,
        "quality_ok": True,
        "require_inject": True,
        "stats": {"inject_count": 0, "g_stream_count": 0, "max_alpha": 0.0},
        "quality": {"garbled_ok": True, "pytest_ok": True, "reasons": []},
        "alpha_0": 0.85,
    }
    session = {
        "n_invocations": 1,
        "invocations": [
            {
                "completion_tokens": 200,
                "assistant_preview": "unicaa01_p30_panic/main.py\n```\ndef parse_ini(t): return {}\n```",
            }
        ],
    }
    assert classify_failure(row, session) == "cast_no_inject"

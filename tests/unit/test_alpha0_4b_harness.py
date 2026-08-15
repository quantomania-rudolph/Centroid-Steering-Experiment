"""Unit tests for prove_cast_alpha0_4b_live helpers (no live server)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.alpha0_gate import select_alpha_0_4b  # noqa: E402
from jlens_qwen.caa.cast_panic_stubs import ELICIT_PREFIX, panic_message  # noqa: E402
from prove_cast_alpha0_4b_live import (  # noqa: E402
    _alpha_tag,
    _binary_midpoint_alpha,
    _next_lower_passing_alpha,
    _snapshot_main,
    detect_oversteer,
)


@pytest.mark.parametrize(
    ("alpha", "tag"),
    [
        (0.80, "a080"),
        (0.85, "a085"),
        (0.90, "a090"),
        (0.95, "a095"),
    ],
)
def test_alpha_tag(alpha: float, tag: str) -> None:
    assert _alpha_tag(alpha) == tag


def test_snapshot_main_writes_under_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import prove_cast_alpha0_4b_live as harness

    monkeypatch.setattr(harness, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "main.py").write_text("def demo():\n    return 1\n", encoding="utf-8")

    dest = _snapshot_main(0.80, "unicaa01_p09_panic", task_dir)
    assert dest == tmp_path / "snapshots" / "a080" / "unicaa01_p09_panic" / "main.py"
    assert dest.is_file()
    assert "def demo" in dest.read_text(encoding="utf-8")


def test_binary_midpoint_alpha_bracket() -> None:
    grid = [0.80, 0.85, 0.90, 0.95]
    rows = [
        {"alpha_0": 0.80, "pass": True, "complete": True},
        {"alpha_0": 0.85, "pass": True, "complete": True},
        {"alpha_0": 0.90, "pass": False, "complete": True},
        {"alpha_0": 0.95, "pass": False, "complete": True},
    ]
    assert _binary_midpoint_alpha(rows, grid) == 0.875


def test_binary_midpoint_alpha_all_pass_returns_none() -> None:
    grid = [0.80, 0.85, 0.90, 0.95]
    rows = [{"alpha_0": a, "pass": True, "complete": True} for a in grid]
    assert _binary_midpoint_alpha(rows, grid) is None


def test_binary_midpoint_alpha_all_fail_returns_none() -> None:
    grid = [0.80, 0.85, 0.90, 0.95]
    rows = [{"alpha_0": a, "pass": False, "complete": True} for a in grid]
    assert _binary_midpoint_alpha(rows, grid) is None


def test_binary_midpoint_alpha_095_pass_no_midpoint_from_that_pair() -> None:
    grid = [0.80, 0.85, 0.90, 0.95]
    rows = [
        {"alpha_0": 0.80, "pass": True, "complete": True},
        {"alpha_0": 0.85, "pass": True, "complete": True},
        {"alpha_0": 0.90, "pass": True, "complete": True},
        {"alpha_0": 0.95, "pass": True, "complete": True},
    ]
    assert _binary_midpoint_alpha(rows, grid) is None


def test_next_lower_passing_alpha() -> None:
    rows = [
        {"alpha_0": 0.80, "pass": True},
        {"alpha_0": 0.85, "pass": True},
        {"alpha_0": 0.90, "pass": False},
    ]
    assert _next_lower_passing_alpha(0.85, rows) == 0.80


def test_detect_oversteer_true_when_inject_worse_than_control() -> None:
    fit_rows = [{"slug": "unicaa01_p09_panic", "quality_ok": False}]
    control_rows = [{"slug": "unicaa01_p09_panic", "quality_ok": True}]
    assert detect_oversteer(fit_rows, control_rows) is True


def test_detect_oversteer_false_when_both_fail_or_pass() -> None:
    fit_rows = [{"slug": "unicaa01_p09_panic", "quality_ok": True}]
    control_rows = [{"slug": "unicaa01_p09_panic", "quality_ok": True}]
    assert detect_oversteer(fit_rows, control_rows) is False

    fit_rows = [{"slug": "unicaa01_p09_panic", "quality_ok": False}]
    control_rows = [{"slug": "unicaa01_p09_panic", "quality_ok": False}]
    assert detect_oversteer(fit_rows, control_rows) is False


def test_select_alpha_0_4b_cap_only_high_passing() -> None:
    rows = [{"alpha_0": 0.90, "pass": True}, {"alpha_0": 0.95, "pass": True}]
    assert select_alpha_0_4b(rows) is None


def test_panic_message_elicit_prefix() -> None:
    with_prefix = panic_message("unicaa01_p30_panic", elicit=True)
    without_prefix = panic_message("unicaa01_p30_panic", elicit=False)
    assert with_prefix.startswith(ELICIT_PREFIX)
    assert without_prefix.startswith("Edit unicaa01_p30_panic/main.py only.")
    assert ELICIT_PREFIX not in without_prefix


def test_argparse_diag_flags_present() -> None:
    import importlib

    harness = importlib.import_module("prove_cast_alpha0_4b_live")
    source = Path(harness.__file__).read_text(encoding="utf-8")
    for flag in ("--no-elicit", "--confirm-no-inject", "--skip-control-stability", "--diag-out"):
        assert flag in source

    argv = [
        "--dry-run",
        "--no-elicit",
        "--confirm-no-inject",
        "--skip-control-stability",
        "--diag-out",
        "artifacts/tmp/diag.json",
    ]
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.85")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--confirm-only", action="store_true")
    ap.add_argument("--confirm-alpha", type=float, default=None)
    ap.add_argument("--confirm-run-slugs", default="")
    ap.add_argument("--rescore-slugs", default="")
    ap.add_argument("--skip-midpoint", action="store_true")
    ap.add_argument("--no-elicit", action="store_true")
    ap.add_argument("--confirm-no-inject", action="store_true")
    ap.add_argument("--skip-control-stability", action="store_true")
    ap.add_argument("--diag-out", type=Path, default=None)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--aider-timeout-s", type=float, default=1200.0)
    args = ap.parse_args(argv)
    assert args.no_elicit is True
    assert args.confirm_no_inject is True
    assert args.skip_control_stability is True
    assert args.diag_out == Path("artifacts/tmp/diag.json")

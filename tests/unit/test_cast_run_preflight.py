"""Unit tests for CAST alpha0 harness preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from jlens_qwen.caa.cast_run_preflight import (
    assert_aider_path_under_qwen_code,
    run_preflight,
)


def test_preflight_purges_repo_root_panic_dir(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen code"
    qwen.mkdir()
    stray = tmp_path / "unicaa01_p09_panic"
    stray.mkdir()
    (stray / "main.py").write_text("stale", encoding="utf-8")

    result = run_preflight(
        repo_root=tmp_path,
        qwen_code=qwen,
        require_stubs=False,
        alphas=[0.8, 0.85, 0.9],
    )
    assert result.ok
    assert not stray.exists()
    assert any("removed" in a for a in result.actions)


def test_preflight_fails_short_alpha_grid(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen code"
    qwen.mkdir()
    result = run_preflight(
        repo_root=tmp_path,
        qwen_code=qwen,
        require_stubs=False,
        alphas=[0.85],
    )
    assert not result.ok
    assert any("alphas grid" in e for e in result.errors)


def test_taskkill_no_running_instance_counts_as_ok() -> None:
    from jlens_qwen.caa.cast_run_preflight import _taskkill_pid

    assert _taskkill_pid(999999, tree=True) is True


def test_list_stray_lab_pids_empty_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    from jlens_qwen.caa.cast_run_preflight import list_stray_lab_pids

    monkeypatch.setattr("jlens_qwen.caa.cast_run_preflight.sys.platform", "linux")
    assert list_stray_lab_pids(repo_root=Path("/tmp")) == []


def test_assert_aider_path_under_qwen_code(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen code"
    task = qwen / "unicaa01_p09_panic"
    task.mkdir(parents=True)
    main_py = task / "main.py"
    main_py.write_text("x", encoding="utf-8")
    assert_aider_path_under_qwen_code(main_py, qwen)
    with pytest.raises(ValueError, match="must be under"):
        assert_aider_path_under_qwen_code(tmp_path / "other.py", qwen)

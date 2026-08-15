"""Unit tests for CAST calibration split manifest v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from cast_calibrate_tau import (  # noqa: E402
    FIT_PANIC_N,
    MANIFEST_VERSION,
    VERIFY_PANIC_N,
    lock_splits,
)


def _fake_panic_run(run_id: str, task_num: str, n_window: int) -> tuple[dict, list]:
    rows = []
    for t in range(20):
        rows.append(
            {
                "t": t,
                "s_cond": 0.5 if t < n_window else 0.0,
                "in_panic_window": t < n_window,
            }
        )
    return (
        {
            "run_id": run_id,
            "kind": "panic",
            "task_slug": f"unicaa01_p{task_num}_panic",
        },
        rows,
    )


def _fake_clean_run(run_id: str) -> tuple[dict, list]:
    rows = [{"t": t, "s_cond": 0.01 * t, "in_panic_window": False} for t in range(30)]
    return ({"run_id": run_id, "kind": "clean", "task_slug": "neutral"}, rows)


def test_lock_splits_v2_assignments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cast_calibrate_tau as mod

    monkeypatch.setattr(mod, "CALIB_DIR", tmp_path)
    clean = [_fake_clean_run(f"clean_{i}") for i in range(10)]
    panic = [_fake_panic_run(f"panic_{i:02d}", f"{i:02d}", n_window=3) for i in range(20)]
    manifest = lock_splits(clean, panic)
    assert manifest["version"] == MANIFEST_VERSION
    assert len(manifest["verify_panic"]) == VERIFY_PANIC_N
    fit_n = int(manifest.get("fit_panic_n") or len(manifest["fit_panic_shape"]))
    assert fit_n == FIT_PANIC_N
    assert len(manifest["fit_panic_shape"]) == FIT_PANIC_N
    overlap = set(manifest["verify_panic"]) & set(manifest["fit_panic_shape"])
    assert not overlap
    for vid in manifest["verify_panic"]:
        ep = next(e for e in manifest["episodes"] if e["run_id"] == vid)
        assert ep["n_window_tokens"] >= 2


def test_lock_splits_requires_enough_verify_eligible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cast_calibrate_tau as mod

    monkeypatch.setattr(mod, "CALIB_DIR", tmp_path)
    clean = [_fake_clean_run(f"clean_{i}") for i in range(10)]
    panic = [_fake_panic_run(f"panic_{i}", f"{i:02d}", n_window=1) for i in range(10)]
    with pytest.raises(RuntimeError, match="verify-eligible"):
        lock_splits(clean, panic)

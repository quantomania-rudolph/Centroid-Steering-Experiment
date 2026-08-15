"""Unit tests for CAST calibration deny list and stream dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.calibration_deny import is_denied_panic_task, load_deny_list
from jlens_qwen.caa.stream_dynamics import panic_window_indices, temporal_resolution


def test_load_deny_list_has_task_slugs():
    meta = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
    deny = load_deny_list(meta)
    assert deny["n_pairs"] == 39
    assert "unicaa01_p01_panic" in deny["task_slugs"]
    assert "01" in deny["task_nums"]


def test_is_denied_panic_task():
    deny = {
        "task_slugs": ["unicaa01_p01_panic"],
        "task_nums": ["01"],
        "dir_pair_slugs": [],
    }
    assert is_denied_panic_task("unicaa01_p01_panic", deny)
    assert not is_denied_panic_task("unicaa01_p05_panic", deny)
    assert not is_denied_panic_task("unicaa01_p01_neutral", deny)


def test_temporal_resolution_enough_tokens():
    tokens = [{"flags": ["B_distress_x"]} for _ in range(20)]
    out = temporal_resolution(tokens)
    assert out["enough_tokens"] is True
    assert "B_distress" in out["metrics"]


def test_panic_window_indices_pointwise_spike():
    tokens = [
        {"flags": [], "B_vocab": {"distress_mass": 0.0, "stub_mass": 0.0}}
        for _ in range(20)
    ]
    tokens[10] = {
        "flags": ["B_distress", "B_stub"],
        "B_vocab": {"distress_mass": 0.12, "stub_mass": 0.09},
    }
    idxs = panic_window_indices(tokens)
    assert idxs == [10]


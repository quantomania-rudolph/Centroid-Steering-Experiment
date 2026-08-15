"""Unit tests for steering vector extraction and capture format adapters."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    _normalize_capture_format,
    compute_steering_vectors,
    save_steering_artifacts,
)


def _fake_capture() -> dict:
    dim = 8
    calm_d = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bad_d = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    shared = [0.5] * dim
    return {
        "layers": [16],
        "d": 3,
        "prompt_len": 10,
        "calm": {16: {"shared": shared, "divergent": calm_d}},
        "bad": {16: {"shared": shared, "divergent": bad_d}},
    }


def test_compute_steering_vectors_orthogonal():
    cap = _fake_capture()
    vecs = compute_steering_vectors(cap)
    assert 16 in vecs.behavior
    assert 16 in vecs.condition
    beh = torch.tensor(vecs.behavior[16])
    cond = torch.tensor(vecs.condition[16])
    assert abs(float(beh.norm()) - 1.0) < 1e-5
    assert abs(float(cond.norm()) - 1.0) < 1e-5
    # behavior = calm - bad, condition = bad - calm
    assert float(torch.dot(beh, cond)) < -0.99


def test_legacy_position_index_adapter():
    legacy = {
        "positions": [100, 101],
        "calm": {"16": {"100": [1.0, 0.0], "101": [1.0, 0.0]}},
        "bad": {"16": {"100": [1.0, 0.0], "101": [0.0, 1.0]}},
    }
    norm = _normalize_capture_format(legacy)
    assert norm["calm"][16]["shared"] == [1.0, 0.0]
    assert norm["bad"][16]["divergent"] == [0.0, 1.0]


def test_save_steering_artifacts(tmp_path: Path):
    vecs = compute_steering_vectors(_fake_capture())
    written = save_steering_artifacts(vecs, tmp_path)
    assert (tmp_path / "behavior_L16.pt").is_file()
    assert (tmp_path / "condition_L16.pt").is_file()
    assert (tmp_path / "metadata.json").is_file()
    payload = torch.load(tmp_path / "behavior_L16.pt", weights_only=False)
    assert payload["kind"] == "behavior"
    assert "u_hat" in payload
    assert written["metadata"]

"""Natural-only aggregation tests (AGG1–AGG9 + sabotage)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    aggregate_steering_vectors,
    compute_steering_vectors,
    normalize_tensor,
    remove_neutral_component,
    save_steering_artifacts,
)


def _cap(calm_d, bad_d, *, source="natural", slug="p", dim=4):
    shared = [0.0] * dim
    return {
        "source": source,
        "pair_slug": slug,
        "task_slug": slug,
        "layers": [16],
        "d": 2,
        "prompt_len": 5,
        "calm": {16: {"shared": shared, "divergent": list(calm_d)}},
        "bad": {16: {"shared": shared, "divergent": list(bad_d)}},
    }


def test_agg1_empty_fail_closed():
    with pytest.raises(ValueError, match="empty"):
        aggregate_steering_vectors([])


def test_agg2_practice_hard_error():
    with pytest.raises(ValueError, match="practice|source"):
        aggregate_steering_vectors([_cap([1, 0, 0, 0], [0, 1, 0, 0], source="practice")])


def test_agg2b_missing_source_fail_closed():
    cap = _cap([1, 0, 0, 0], [0, 1, 0, 0])
    cap.pop("source")
    with pytest.raises(ValueError, match="source"):
        aggregate_steering_vectors([cap])


def test_agg3_mean_then_normalize():
    c1 = _cap([2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], slug="a")
    c2 = _cap([0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], slug="b")
    # diffs: [2,0,0,0] and [0,2,0,0] → mean [1,1,0,0] → normalize
    out = aggregate_steering_vectors([c1, c2], require_neutral_for_production=False)
    beh = torch.tensor(out.behavior[16])
    expected = normalize_tensor(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(beh, expected, atol=1e-6)
    assert out.n_pairs == 2
    assert out.pair_slugs == ["a", "b"]


def test_agg4_anti_average_of_unit_vectors():
    # Unequal magnitudes: mean-then-normalize ≠ normalize-then-mean
    c1 = _cap([10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], slug="a")
    c2 = _cap([0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], slug="b")
    out = aggregate_steering_vectors([c1, c2], require_neutral_for_production=False)
    u1 = normalize_tensor(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    u2 = normalize_tensor(torch.tensor([0.0, 2.0, 0.0, 0.0]))
    wrong = normalize_tensor((u1 + u2) / 2)
    right = torch.tensor(out.behavior[16])
    assert not torch.allclose(right, wrong, atol=1e-5)
    # Correct: mean of raw diffs [5, 1, 0, 0] then normalize
    expected = normalize_tensor(torch.tensor([5.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(right, expected, atol=1e-5)


def test_agg5_metadata_fields():
    out = aggregate_steering_vectors(
        [_cap([1, 0, 0, 0], [0, 1, 0, 0])],
        require_neutral_for_production=False,
    )
    assert out.source == "natural"
    assert out.n_pairs == 1
    assert out.production_ready is False  # AGG9


def test_agg8_production_requires_neutral(tmp_path: Path):
    out = aggregate_steering_vectors(
        [_cap([1, 0, 0, 0], [0, 1, 0, 0])],
        require_neutral_for_production=True,
    )
    assert out.neutral_applied is False
    with pytest.raises(ValueError, match="neutral"):
        save_steering_artifacts(out, tmp_path, require_neutral=True)


def test_agg_neutral_cleanup():
    neutral = {16: [1.0, 0.0, 0.0, 0.0]}
    # calm-bad = [1,1,0,0]; remove x-component → [0,1,0,0]
    out = aggregate_steering_vectors(
        [_cap([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])],
        neutral=neutral,
        require_neutral_for_production=True,
    )
    assert out.neutral_applied is True
    beh = torch.tensor(out.behavior[16])
    assert abs(float(beh[0])) < 1e-5
    assert abs(float(beh[1]) - 1.0) < 1e-5


def test_id10_n1_matches_compute():
    cap = _cap([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
    single = compute_steering_vectors(cap)
    pooled = aggregate_steering_vectors([cap], require_natural=False, require_neutral_for_production=False)
    assert torch.allclose(
        torch.tensor(single.behavior[16]),
        torch.tensor(pooled.behavior[16]),
        atol=1e-5,
    )


def test_aggregate_gate(tmp_path: Path):
    payload = write_gate(
        tmp_path / "caa_aggregate_natural_only_gate.json",
        gate="caa_aggregate_natural_only",
        pass_=True,
        checks=[{"id": f"AGG{i}", "pass": True, "detail": "ok"} for i in range(1, 10)],
        sabotage=[{"id": "AGG-S1", "pass": True, "detail": "label swap deferred to identity"}],
    )
    assert payload["pass"]

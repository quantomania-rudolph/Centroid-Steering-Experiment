"""Vector identity proofs ID1–ID10 + sabotage (fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    aggregate_steering_vectors,
    compute_steering_vectors,
    normalize_tensor,
    remove_neutral_component,
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _cap(calm, bad, layer=16):
    dim = len(calm)
    shared = [0.0] * dim
    return {
        "source": "natural",
        "layers": [layer],
        "d": 1,
        "prompt_len": 2,
        "pair_slug": "fix",
        "calm": {layer: {"shared": shared, "divergent": list(calm)}},
        "bad": {layer: {"shared": shared, "divergent": list(bad)}},
    }


def test_id1_id2_separation():
    calm = [1.0, 0.1, 0.0, 0.0]
    bad = [0.1, 1.0, 0.0, 0.0]
    vecs = compute_steering_vectors(_cap(calm, bad))
    u_beh = torch.tensor(vecs.behavior[16])
    u_cond = torch.tensor(vecs.condition[16])
    h_c = torch.tensor(calm)
    h_b = torch.tensor(bad)
    assert _cos(h_c, u_beh) - _cos(h_b, u_beh) >= 0.10
    assert _cos(h_b, u_cond) - _cos(h_c, u_cond) >= 0.10


def test_id5_opposing_single_pair():
    vecs = compute_steering_vectors(_cap([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]))
    cos = _cos(torch.tensor(vecs.behavior[16]), torch.tensor(vecs.condition[16]))
    assert cos <= -0.85


def test_id8_swap_labels_fails_separation():
    calm = [1.0, 0.0, 0.0, 0.0]
    bad = [0.0, 1.0, 0.0, 0.0]
    # Train on correct labels
    vecs = compute_steering_vectors(_cap(calm, bad))
    u_beh = torch.tensor(vecs.behavior[16])
    u_cond = torch.tensor(vecs.condition[16])
    # Evaluate on swapped — ID1/ID2 must fail
    h_c = torch.tensor(bad)  # swapped
    h_b = torch.tensor(calm)
    id1 = _cos(h_c, u_beh) - _cos(h_b, u_beh) >= 0.10
    id2 = _cos(h_b, u_cond) - _cos(h_c, u_cond) >= 0.10
    assert id1 is False
    assert id2 is False


def test_id6_id7_neutral_not_aligned():
    u_n = normalize_tensor(torch.tensor([0.0, 0.0, 1.0, 0.0]))
    vecs = compute_steering_vectors(_cap([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]))
    assert abs(_cos(u_n, torch.tensor(vecs.behavior[16]))) < 0.50
    assert abs(_cos(u_n, torch.tensor(vecs.condition[16]))) < 0.50


def test_id9_cleanup_reduces_projection():
    u_n = normalize_tensor(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    v = torch.tensor([5.0, 1.0, 0.0, 0.0])
    before = abs(float(torch.dot(v, u_n) / v.norm()))
    cleaned = remove_neutral_component(v, u_n)
    after = abs(float(torch.dot(cleaned, u_n) / (cleaned.norm() + 1e-12)))
    assert after <= before * 0.05 + 1e-6


def test_id10_pooled_n1():
    cap = _cap([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
    a = compute_steering_vectors(cap)
    b = aggregate_steering_vectors(
        [cap], require_natural=False, require_neutral_for_production=False
    )
    assert torch.allclose(torch.tensor(a.behavior[16]), torch.tensor(b.behavior[16]), atol=1e-5)


def test_identity_gate(tmp_path: Path):
    payload = write_gate(
        tmp_path / "caa_vector_identity_gate.json",
        gate="caa_vector_identity",
        pass_=True,
        checks=[{"id": f"ID{i}", "pass": True, "detail": "fixture"} for i in range(1, 11)],
        sabotage=[
            {"id": "ID-S1", "pass": True, "detail": "swap fails ID1-4"},
            {"id": "ID-S2", "pass": True, "detail": "neutral!=cond"},
        ],
    )
    assert payload["pass"]

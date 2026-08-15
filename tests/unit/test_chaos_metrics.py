"""Unit tests for A–G chaos metrics (golden vectors)."""

from __future__ import annotations

import math

import pytest
import torch

from jlens_qwen.chaos.bus import ChaosBus
from jlens_qwen.chaos.metrics_a import (
    distribution_shape,
    entropy_from_probs,
    gini,
    margin,
    tail_mass,
)
from jlens_qwen.chaos.metrics_b import masses_from_topk, transport_masses
from jlens_qwen.chaos.metrics_c import trajectory_from_sequence
from jlens_qwen.chaos.metrics_d import cross_position, js_divergence
from jlens_qwen.chaos.metrics_e import layer_curve_metrics
from jlens_qwen.chaos.metrics_f import lens_disagreement, topm_overlap
from jlens_qwen.chaos.metrics_g import sparse_residual
from jlens_qwen.jspace_decompose import gradient_pursuit


def test_a_entropy_uniform():
    p = torch.ones(5) / 5
    h = entropy_from_probs(p)
    assert abs(h - math.log(5)) < 1e-5


def test_a_margin_and_tail():
    p = torch.tensor([0.7, 0.2, 0.05, 0.03, 0.02])
    assert abs(margin(p) - 0.5) < 1e-6
    # ranks 6.. empty → 0
    assert tail_mass(p, from_rank=6) == 0.0
    p20 = torch.tensor([0.5] + [0.5 / 19] * 19)
    assert abs(tail_mass(p20, from_rank=6) - (0.5 / 19) * 15) < 1e-5


def test_a_gini_delta():
    # all mass on one → gini ~ 1 - 1/k for sorted formula; check concentrated > uniform
    peaked = torch.tensor([0.99] + [0.01 / 19] * 19)
    uniform = torch.ones(20) / 20
    assert gini(peaked) > gini(uniform)


def test_a_distribution_shape_from_logits():
    logits = torch.tensor([5.0, 1.0, 0.0, -1.0, -2.0])
    out = distribution_shape(logits, k=5, full_entropy=True)
    assert out["entropy"] > 0
    assert out["margin"] > 0
    assert "kurtosis" in out


def test_b_masses_hand_calc():
    # ids 0,1,2 with probs 0.5,0.3,0.2; code={0,1}, distress={2}
    ids = torch.tensor([0, 1, 2])
    probs = torch.tensor([0.5, 0.3, 0.2])
    masks = {
        "code": torch.tensor([True, True, False, False]),
        "distress": torch.tensor([False, False, True, False]),
        "stub": torch.tensor([False, False, False, False]),
        "junk": torch.tensor([False, False, False, False]),
    }
    m = masses_from_topk(ids, probs, masks)
    assert abs(m["code_mass"] - 0.8) < 1e-6
    assert abs(m["distress_mass"] - 0.2) < 1e-6
    assert m["stub_mass"] == 0.0


def test_b_transport_masses_sum_one():
    d = 8
    h = torch.randn(d)
    P = torch.randn(4, d)
    m = transport_masses(h, P)
    s = sum(m.values())
    assert abs(s - 1.0) < 1e-5


def test_c_oscillation_flip():
    # A-B-A-B-A
    out = trajectory_from_sequence([1, 2, 1, 2, 1], window=5)
    assert out["flip_rate"] > 0.5
    assert out["oscillation"] > 0.0
    assert out["unique_ratio"] == 2 / 5


def test_d_js_symmetric_bounded():
    ids_a = torch.tensor([1, 2, 3])
    pa = torch.tensor([0.5, 0.3, 0.2])
    ids_b = torch.tensor([3, 4, 5])
    pb = torch.tensor([0.4, 0.4, 0.2])
    js_ab = js_divergence(ids_a, pa, ids_b, pb)
    js_ba = js_divergence(ids_b, pb, ids_a, pa)
    assert abs(js_ab - js_ba) < 1e-6
    assert 0 <= js_ab <= math.log(2) + 1e-6
    # identical → ~0
    js0 = js_divergence(ids_a, pa, ids_a, pa)
    assert js0 < 1e-6


def test_d_cross_position():
    ids = torch.tensor([1, 2])
    p = torch.tensor([0.6, 0.4])
    out = cross_position(
        entropy_a=1.0,
        entropy_b=2.0,
        code_mass_a=0.5,
        code_mass_b=0.2,
        ids_a=ids,
        probs_a=p,
        ids_b=ids,
        probs_b=p,
    )
    assert abs(out["delta_H"] - 1.0) < 1e-6
    assert abs(out["delta_M_code"] - (-0.3)) < 1e-6


def test_e_cliff_at_l16():
    series = {L: 0.1 for L in range(8, 27)}
    series[16] = 0.1
    series[17] = 0.9  # big jump at L16→L17 → cliff_layer=16
    out = layer_curve_metrics(series)
    assert out["cliff_layer"] == 16
    assert abs(out["cliff_mag"]) > 0.5


def test_f_disjoint_overlap_zero():
    assert topm_overlap([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], m=5) == 0.0
    d = lens_disagreement([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], m=5)
    assert d["overlap"] == 0.0


def test_f_identical_overlap_one():
    ids = [10, 20, 30, 40, 50]
    assert abs(topm_overlap(ids, ids, m=5) - 1.0) < 1e-9


def test_f_alarm_sampling_agreement():
    from jlens_qwen.chaos.metrics_f import f_alarm_from_metrics

    ids = [1, 2, 3, 4, 5]
    agree = lens_disagreement(ids, ids)
    assert agree["overlap"] == 1.0
    assert not f_alarm_from_metrics(
        {"overlap_sample": agree["overlap"], "overlap": agree["overlap"]}
    )


def test_f_alarm_disjoint_fires():
    from jlens_qwen.chaos.metrics_f import f_alarm_from_metrics

    disagree = lens_disagreement([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert disagree["overlap"] == 0.0
    assert f_alarm_from_metrics({"overlap_sample": disagree["overlap"]})


def test_f_legacy_naive_only_no_alarm():
    from jlens_qwen.chaos.metrics_f import f_alarm_from_metrics

    assert not f_alarm_from_metrics({"overlap": 0.0})
    assert not f_alarm_from_metrics({"overlap": 0.6})


def test_g_atom_low_relmse():
    # h is exactly a dictionary direction after transport-like setup:
    # use decompose path: dictionary rows are W_U @ J; pick h aligned with one atom
    d = 16
    V = 64
    torch.manual_seed(0)
    W_U = torch.randn(V, d)
    J = torch.eye(d)
    # pick atom 0 in transported space: h such that W_U[0] @ J @ h is large
    # simplest: set h = J.T @ W_U[0] normalized → strong correlation with atom 0
    h = (J.T @ W_U[0].float()).float()
    out = sparse_residual(h, W_U, J, k=5, max_dict=32)
    assert out["relMSE"] < 0.05 or out["support_size"] >= 1
    # stronger: gradient_pursuit on dictionary atom itself
    dictionary = W_U[:32] @ J
    idx, coef, rel = gradient_pursuit(dictionary[0], dictionary, k=5)
    assert rel < 0.05


def test_chaos_bus_flags_never_dropped_on_downsample():
    bus = ChaosBus()
    bus.begin_invocation("s", "i", device="cpu")
    # 600 tokens, only step 100 flagged via high entropy payload
    for step in range(600):
        payload = {
            "A_vocab": {"entropy": 3.5 if step == 100 else 0.5, "margin": 0.5},
            "B_vocab": {
                "code_mass": 0.5,
                "distress_mass": 0.0,
                "stub_mass": 0.0,
                "junk_mass": 0.0,
            },
            "C_vocab": {"flip_rate": 0.1, "oscillation": 0.0, "settling_tau": -1},
        }
        bus.record_token(step, step % 50, payload)
    rec = bus.finalize()
    assert len(rec["tokens"]) < 600  # downsampled
    flagged = [t for t in rec["tokens"] if t.get("flags")]
    assert any(t["step"] == 100 for t in flagged)
    assert rec["composites"]["PLDS"] >= 0

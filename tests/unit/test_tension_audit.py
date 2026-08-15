"""Unit tests for CAST chaos vs u_cond tension audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.cast_math import gamma
from jlens_qwen.caa.stream_dynamics import panic_window_indices
from jlens_qwen.caa.tension_audit import (
    C_PLUS_G_MINUS_CAP,
    classify_token_tension,
    panic_window_kind_per_step,
    tension_gate_passes,
)


def test_classify_c_plus_g_minus() -> None:
    row = {"t": 0, "s_cond": -0.1, "in_panic_window": True}
    tr = classify_token_tension(
        row,
        tau=0.02,
        sigma=0.001,
        tau_hard=-0.05,
        g_stream=True,
        window_kind="pointwise",
    )
    assert tr.bucket == "C_plus_G_minus"
    assert tr.gamma_val < 0.5


def test_classify_agree_plus() -> None:
    row = {"t": 0, "s_cond": 0.5, "in_panic_window": True}
    tr = classify_token_tension(
        row,
        tau=0.02,
        sigma=0.001,
        tau_hard=-0.05,
        g_stream=True,
        window_kind="pointwise",
    )
    assert tr.bucket == "agree_plus"
    assert tr.gamma_val >= 0.5


def test_panic_window_kind_pointwise_first() -> None:
    tokens = [
        {
            "flags": ["B_distress", "B_stub"],
            "B_vocab": {"distress_mass": 0.6, "stub_mass": 0.6},
        }
        for _ in range(12)
    ]
    kinds = panic_window_kind_per_step(tokens, max_new_tokens=12)
    assert kinds[0] == "pointwise"


def test_weak_pointwise_spike_excluded_from_window() -> None:
    calm = {
        "flags": [],
        "B_vocab": {"distress_mass": 0.0, "stub_mass": 0.0},
    }
    tokens = [calm for _ in range(16)]
    tokens[3] = {
        "flags": ["B_distress", "B_stub"],
        "B_vocab": {"distress_mass": 0.33, "stub_mass": 0.33},
    }
    tokens[10] = {
        "flags": ["B_distress", "B_stub"],
        "B_vocab": {"distress_mass": 0.99, "stub_mass": 0.99},
    }
    tokens[11] = {
        "flags": ["B_distress", "B_stub"],
        "B_vocab": {"distress_mass": 0.99, "stub_mass": 0.99},
    }
    idxs = panic_window_indices(tokens, max_new_tokens=len(tokens))
    assert 3 not in idxs
    assert 10 in idxs and 11 in idxs


def test_tension_gate_passes_on_clean_audit() -> None:
    audit = {
        "corpus_c_plus_g_minus_rate": 0.05,
        "low_gamma_thick_episodes": 1,
        "clean_fpr_gamma_high": 0.02,
        "thresholds": {
            "c_plus_g_minus_cap": C_PLUS_G_MINUS_CAP,
            "max_low_gamma_episodes": 3,
        },
    }
    ok, checks = tension_gate_passes(audit)
    assert ok
    assert len(checks) == 3


def test_gamma_floor_matches_cast_math() -> None:
    g = gamma(0.5, tau=0.02, sigma=0.001, tau_hard=-0.05)
    assert g >= 0.99

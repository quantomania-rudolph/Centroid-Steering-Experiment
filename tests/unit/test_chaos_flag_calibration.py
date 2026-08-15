"""Strict flag calibration — calm vs distressed payloads must separate."""

from __future__ import annotations

from jlens_qwen.chaos.flags import evaluate_flags


def _calm_token() -> dict:
    """Synthetic calm codegen token: confident, code-heavy, stable class."""
    return {
        "A_vocab": {"entropy": 0.8, "margin": 0.45},
        "B_vocab": {
            "code_mass": 0.35,
            "distress_mass": 0.0,
            "stub_mass": 0.0,
            "junk_mass": 0.0,
        },
        "C_vocab": {
            "flip_rate": 0.5,
            "oscillation": 0.1,
            "unique_ratio": 0.6,
            "settling_tau": 20.0,
            "window_full": 1.0,
        },
    }


def _distressed_token() -> dict:
    """Synthetic distressed: flat, low code, oscillating."""
    return {
        "A_vocab": {"entropy": 2.8, "margin": 0.05},
        "B_vocab": {
            "code_mass": 0.01,
            "distress_mass": 0.2,
            "stub_mass": 0.15,
            "junk_mass": 0.0,
        },
        "C_vocab": {
            "flip_rate": 0.95,
            "oscillation": 0.5,
            "unique_ratio": 0.95,
            "settling_tau": -1.0,
            "window_full": 1.0,
        },
        "D_boundary": True,
        "D": {"delta_H": 1.2, "js_div": 0.5, "delta_M_code": -0.2},
    }


def test_calm_token_minimal_flags():
    flags = evaluate_flags(_calm_token())
    assert "C_thrash" not in flags
    assert "B_distress" not in flags
    assert "D_focus_break" not in flags


def test_distressed_token_fires_multiple():
    flags = evaluate_flags(_distressed_token())
    assert "A_low_margin" in flags or "A_high_entropy" in flags
    assert "B_distress" in flags or "B_stub" in flags
    assert "C_thrash" in flags or "C_oscillation" in flags
    assert "D_focus_break" in flags


def test_rolling_d_does_not_fire_without_boundary():
    payload = _calm_token()
    payload["D"] = {"delta_H": 2.0, "js_div": 0.9}
    flags = evaluate_flags(payload)
    assert "D_focus_break" not in flags


def test_boundary_d_fires():
    payload = _calm_token()
    payload["D_boundary"] = True
    payload["D"] = {"delta_H": 1.0, "js_div": 0.1}
    flags = evaluate_flags(payload)
    assert "D_focus_break" in flags


def test_normal_flip_alone_not_thrash():
    """High flip during codegen without oscillation must not alarm."""
    payload = _calm_token()
    payload["C_vocab"] = {
        "flip_rate": 0.99,
        "oscillation": 0.05,
        "unique_ratio": 0.95,
        "settling_tau": -1.0,
        "window_full": 1.0,
    }
    flags = evaluate_flags(payload)
    assert "C_thrash" not in flags
    assert "C_oscillation" not in flags


def test_b_low_code_fires():
    flags = evaluate_flags(_distressed_token())
    assert "B_low_code" in flags


def test_b_low_code_calm_does_not_fire():
    flags = evaluate_flags(_calm_token())
    assert "B_low_code" not in flags


def test_oscillation_thrash_fires():
    payload = _calm_token()
    payload["C_vocab"] = {
        "flip_rate": 0.8,
        "oscillation": 0.5,
        "unique_ratio": 0.5,
        "settling_tau": -1.0,
        "window_full": 1.0,
    }
    flags = evaluate_flags(payload)
    assert "C_thrash" in flags


def test_g_threshold_in_thresh():
    from jlens_qwen.chaos.flags import THRESH

    assert THRESH["G_high_relmse"] == 0.05

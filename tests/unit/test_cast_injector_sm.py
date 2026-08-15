"""CastInjectorState episode SM."""

from __future__ import annotations

from pathlib import Path

import pytest

from jlens_qwen.caa.cast_injector import CastInjectorState, CastLiveSession, MAX_INJECT_TOKENS

ROOT = Path(__file__).resolve().parents[2]
CAA = ROOT / "artifacts" / "caa" / "UNI_CAA01"


def test_cap_at_64_tokens() -> None:
    sm = CastInjectorState()
    for _ in range(MAX_INJECT_TOKENS):
        assert sm.cap_alpha(0.5) == 0.5
        sm.note_inject_applied(True)
    assert sm.cap_alpha(0.5) == 0.0


def test_early_stop_when_masses_drop() -> None:
    sm = CastInjectorState()
    sm.note_inject_applied(True)
    assert sm.should_stop_early(0.01, 0.01) is True
    assert sm.cap_alpha(0.8) == 0.0


def test_shadow_never_sets_pending_alpha() -> None:
    session = CastLiveSession(
        CAA / "metadata.json",
        CAA / "condition_L16.pt",
        CAA / "behavior_L16.pt",
        shadow=True,
        inject=False,
    )
    session.reset(max_new_tokens=128)
    row = session.process_token(
        0,
        distress_mass=0.2,
        stub_mass=0.2,
        flags=["B_distress", "B_stub"],
        recipe_id="B_STUB_DISTRESS",
        severity="critical",
        h_l16=None,
        max_new_tokens=128,
    )
    assert row["alpha_eff"] >= 0.0
    assert session.controller.pending_alpha == 0.0
    assert "gamma_zeroed_by_tau_hard" in row


def test_inject_sets_pending_for_next_forward() -> None:
    import torch

    from jlens_qwen.caa.s_cond_replay import load_u_cond

    session = CastLiveSession(
        CAA / "metadata.json",
        CAA / "condition_L16.pt",
        CAA / "behavior_L16.pt",
        shadow=True,
        inject=True,
    )
    session.reset(max_new_tokens=128)
    u = load_u_cond(CAA / "condition_L16.pt")
    h = (u * 2.0).view(1, 1, -1)
    for i in range(10):
        session.process_token(
            i,
            distress_mass=0.0,
            stub_mass=0.0,
            flags=[],
            recipe_id=None,
            severity=None,
            h_l16=None,
            max_new_tokens=128,
        )
    row = session.process_token(
        10,
        distress_mass=0.6,
        stub_mass=0.6,
        flags=["B_distress", "B_stub"],
        recipe_id="B_STUB_DISTRESS",
        severity="warn",
        h_l16=h,
        max_new_tokens=128,
    )
    assert row["g_stream"] == 1
    assert row["alpha_eff"] > 0
    assert session.controller.pending_alpha > 0

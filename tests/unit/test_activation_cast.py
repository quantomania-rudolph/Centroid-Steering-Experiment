"""ActivationCAST s_cond/gamma vs offline replay."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jlens_qwen.caa.activation_cast import ActivationCAST
from jlens_qwen.caa.cast_math import gamma, load_cast_thresholds
from jlens_qwen.caa.s_cond_replay import load_replay_jsonl, load_u_cond

ROOT = Path(__file__).resolve().parents[2]
CAA = ROOT / "artifacts" / "caa" / "UNI_CAA01"
PANIC = CAA / "calibration" / "panic" / "panic__unicaa01_corpus_p09_thread_pool_panic_latest__inv1.jsonl"


@pytest.fixture
def cast() -> ActivationCAST:
    return ActivationCAST(CAA / "metadata.json", CAA / "condition_L16.pt")


def test_evaluate_none_returns_zeros(cast: ActivationCAST) -> None:
    out = cast.evaluate(None)
    assert out["s_cond"] == 0.0
    assert out["gamma"] == 0.0


def test_gamma_tau_hard_on_random_h(cast: ActivationCAST) -> None:
    u = load_u_cond(CAA / "condition_L16.pt")
    h = torch.randn(1, 1, u.numel())
    out = cast.evaluate(h)
    thresholds = load_cast_thresholds(CAA / "metadata.json")
    expected = gamma(out["s_cond"], thresholds["tau"], thresholds["sigma"], thresholds["tau_hard"])
    assert abs(float(out["gamma"]) - expected) < 1e-6


def test_panic_spike_high_gamma(cast: ActivationCAST) -> None:
    _meta, records = load_replay_jsonl(PANIC)
    spike = next(r for r in records if r.get("in_panic_window"))
    sc = float(spike["s_cond"])
    thresholds = load_cast_thresholds(CAA / "metadata.json")
    g = gamma(sc, thresholds["tau"], thresholds["sigma"], thresholds["tau_hard"])
    assert g >= 0.7
    assert sc > thresholds["tau"] + 0.05

"""Unit tests for neutral capture purity N1–N10 + sabotage."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from caa_error_combo_study import score_text  # noqa: E402
from jlens_qwen.caa.neutral_capture import (  # noqa: E402
    NEUTRAL_TASKS,
    capture_neutral_gen_start,
    verify_gen_start_tokenization,
)
from jlens_qwen.caa.neutral_purity import run_neutral_purity_checks  # noqa: E402
from jlens_qwen.caa.steering_vectors import normalize_tensor, remove_neutral_component  # noqa: E402
from jlens_qwen.chaos.constants import PROBE_LAYERS  # noqa: E402
from prefix_divergence_tracer import eligible_for_panic_trace  # noqa: E402


def test_n1_neutral_task_texts_clean():
    for task in NEUTRAL_TASKS:
        s = score_text(task["user_prompt"] + " " + task["assistant_start"])
        assert s["stub_score"] == 0
        assert s["distress_score"] == 0


def test_n2_tasks_are_clean_kata_only():
    slugs = {t["slug"] for t in NEUTRAL_TASKS}
    assert slugs == {"chaos_proof_clean", "neutral_topo_sort"}
    assert "chaos_proof_meta_wipe" not in slugs
    assert len(NEUTRAL_TASKS) >= 2


def test_n_s2_requires_two_tasks():
    with pytest.raises(ValueError, match="≥2"):
        capture_neutral_gen_start(None, tasks=[NEUTRAL_TASKS[0]])  # type: ignore[arg-type]


def test_n_s1_panic_text_fails_n1():
    s = score_text("Sorry pip install raise NotImplementedError")
    assert s["stub_score"] >= 1 or s["distress_score"] >= 1


def _mock_bundle(dim: int = 8):
    captured: dict[int, dict[int, list[float]]] = {}

    def fake_capture(bundle, input_ids, positions, layers=None):
        out = {}
        for L in layers or PROBE_LAYERS:
            out[L] = {}
            for pos in positions:
                vec = [float(L) / 100.0] * dim
                vec[0] = float(pos) / 1000.0
                out[L][pos] = vec
        return out

    tok = MagicMock()
    tok.apply_chat_template.return_value = [1, 2, 3, 4, 5]
    tok.encode.side_effect = lambda text, add_special_tokens=False: [42] if "python" in text else [99]

    bundle = SimpleNamespace(tokenizer=tok, device="cpu")
    return bundle, fake_capture


def test_capture_neutral_gen_start_mock_bundle(monkeypatch):
    bundle, fake_capture = _mock_bundle()
    monkeypatch.setattr(
        "jlens_qwen.caa.neutral_capture.capture_residuals_at_positions",
        fake_capture,
    )
    monkeypatch.setattr(
        "jlens_qwen.caa.neutral_capture.verify_gen_start_tokenization",
        lambda *a, **k: (True, "ok"),
    )
    result = capture_neutral_gen_start(bundle)
    assert result.n_tasks == 2
    assert set(result.neutral.keys()) == set(PROBE_LAYERS)
    for L in PROBE_LAYERS:
        norm = float(torch.tensor(result.neutral[L]).norm().item())
        assert abs(norm - 1.0) < 1e-5


def test_neutral_purity_gate_unit_mode_fails_without_live():
    checks, sab = run_neutral_purity_checks(
        None,
        caa_root=ROOT / "artifacts" / "caa" / "UNI_CAA01",
        score_text_fn=score_text,
        eligible_for_panic_trace_fn=eligible_for_panic_trace,
        capture_neutral_gen_start_fn=capture_neutral_gen_start,
        bundle=None,
    )
    assert any(c["id"].startswith("N3_") and c["pass"] is False for c in checks)
    assert any(s["id"] == "N-S2" for s in sab)


def test_remove_neutral_component_geometry():
    u_n = normalize_tensor(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    v = torch.tensor([3.0, 4.0, 0.0, 0.0])
    cleaned = remove_neutral_component(v, u_n)
    proj_after = abs(float(torch.dot(cleaned, u_n) / (cleaned.norm() + 1e-12)))
    assert proj_after < 0.05

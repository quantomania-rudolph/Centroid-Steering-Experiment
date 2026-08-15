"""Unit tests for natural miner pairing + registry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.calm_tail_registry import calm_tail_filename, find_calm_tail  # noqa: E402
from jlens_qwen.caa.kv_stamp import read_global_kv_live_gate  # noqa: E402
from jlens_qwen.caa.miner_pairing import (  # noqa: E402
    build_clean_chaos_from_token_ids,
    check_d_coherence,
    extract_d_from_lcp,
)


def test_extract_d_from_lcp():
    calm = [1, 2, 3, 10, 11]
    bad = [1, 2, 3, 99, 88]
    assert extract_d_from_lcp(calm, bad) == 3


def test_d_coherence_pass_and_fail():
    ok = check_d_coherence(55, 55)
    assert ok.passed
    bad = check_d_coherence(10, 55, debounce=15)
    assert not bad.passed


def test_clean_chaos_zero_rates():
    chaos = build_clean_chaos_from_token_ids([1, 2, 3])
    assert chaos["divergence_hints"]["b_distress_rate"] == 0.0
    assert chaos["divergence_hints"]["b_stub_rate"] == 0.0


def test_calm_tail_registry_lookup(tmp_path: Path):
    import json

    name = calm_tail_filename("live_panic_demo_scheduler", "chatcmpl-abc123", 55)
    p = tmp_path / name
    p.write_text("def ok(): pass", encoding="utf-8")
    meta = {
        "status": "admitted",
        "pipeline_version": 5,
        "repaired": False,
        "bridge_used": False,
        "sanitized": False,
    }
    p.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    found = find_calm_tail("live_panic_demo_scheduler", "chatcmpl-abc123", 55, root=tmp_path)
    assert found == p


def test_calm_tail_fuzzy_requires_matching_d(tmp_path: Path):
    import json

    # Wrong d should not match via fuzzy fallback.
    wrong = tmp_path / calm_tail_filename("live_panic_demo_scheduler", "chatcmpl-abc123xyz", 99)
    wrong.write_text("def ok(): pass", encoding="utf-8")
    meta = {
        "status": "admitted",
        "pipeline_version": 5,
        "repaired": False,
        "bridge_used": False,
        "sanitized": False,
    }
    wrong.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert find_calm_tail("live_panic_demo_scheduler", "chatcmpl-abc123xyz", 55, root=tmp_path) is None


def test_pool_eligible_on_disk_requires_admission(tmp_path: Path):
    from jlens_qwen.caa.natural_miner import _pool_eligible_on_disk

    d = tmp_path / "pair"
    d.mkdir()
    (d / "activations_at_d.pt").write_bytes(b"x")
    assert not _pool_eligible_on_disk(d)
    (d / "admission_gate.json").write_text(
        '{"pass": true, "pool_eligible": true, "source": "natural"}',
        encoding="utf-8",
    )
    assert _pool_eligible_on_disk(d)


def test_cap7_rejects_nan():
    from jlens_qwen.caa.admission import verify_capture_integrity
    from jlens_qwen.chaos.constants import PROBE_LAYERS

    layers = list(PROBE_LAYERS)
    calm = {L: {"shared": [1.0, 2.0], "divergent": [float("nan"), 1.0]} for L in layers}
    bad = {L: {"shared": [1.0, 2.0], "divergent": [0.0, 1.0]} for L in layers}
    cap = {
        "d": 3,
        "prompt_len": 10,
        "positions": {"shared": 12, "divergent": 13},
        "layers": layers,
        "calm": calm,
        "bad": bad,
    }
    out = verify_capture_integrity(cap, d=3, prompt_len=10)
    assert not out["pass"]
    assert any(c["id"] == "CAP7_finite_activations" and not c["pass"] for c in out["checks"])
def test_kv_live_gate_readable():
    gate = read_global_kv_live_gate(ROOT / "artifacts" / "gates" / "kv_equivalence_live_gate.json")
    assert "pass" in gate

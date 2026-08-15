"""Structural KV equivalence unit tests (KV1–KV7). No numerical claim without model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.activation_capture import _clone_past_key_values  # noqa: E402
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.kv_equivalence import (  # noqa: E402
    compare_captures,
    extract_positions,
    structural_prefix_guard,
)
from jlens_qwen.chaos.constants import PROBE_LAYERS  # noqa: E402


def test_clone_past_key_values_independent():
    t = torch.randn(1, 4, 8, 16)
    past = ((t,),)
    c1 = _clone_past_key_values(past)
    c2 = _clone_past_key_values(past)
    c1[0][0][0, 0, 0, 0] = 999.0
    assert float(c2[0][0][0, 0, 0, 0]) != 999.0
    assert float(past[0][0][0, 0, 0, 0]) != 999.0


def test_kv1_prefix_guard_raises():
    with pytest.raises(ValueError, match="prefix"):
        structural_prefix_guard([1, 2, 3], [1, 9, 3], d=2)


def test_kv2_positions():
    pos = extract_positions(prompt_len=78, d=77)
    assert pos["shared"] == 154
    assert pos["divergent"] == 155
    pos0 = extract_positions(prompt_len=10, d=0)
    assert pos0["shared"] == 9
    assert pos0["divergent"] == 10


def test_kv3_kv4_kv6_kv7_compare_structure():
    layers = list(PROBE_LAYERS)
    shared = [0.1] * 4
    calm_d = [1.0, 0.0, 0.0, 0.0]
    bad_d = [0.0, 1.0, 0.0, 0.0]
    kv = {
        "layers": layers,
        "prompt_len": 10,
        "d": 3,
        "positions": extract_positions(10, 3),
        "capture_method": "shared_kv_prefill",
        "calm": {L: {"shared": shared, "divergent": calm_d} for L in layers},
        "bad": {L: {"shared": shared, "divergent": bad_d} for L in layers},
    }
    # Identical full for structural pass of non-numerical checks;
    # numerical will also pass when identical.
    full = dict(kv)
    full["capture_method"] = "full_forward"
    results = compare_captures(full, kv)
    by_id = {r.id: r for r in results}
    assert by_id["KV2_positions"].passed
    assert by_id["KV3_shared_identical"].passed
    assert by_id["KV4_layers"].passed
    assert by_id["KV6_capture_method"].passed
    assert by_id["KV7_databus_positions"].passed


def test_kv3_shared_mismatch_fails():
    layers = [16]
    kv = {
        "layers": layers,
        "prompt_len": 2,
        "d": 1,
        "positions": extract_positions(2, 1),
        "capture_method": "shared_kv_prefill",
        "calm": {16: {"shared": [1.0], "divergent": [1.0]}},
        "bad": {16: {"shared": [0.0], "divergent": [0.0]}},
    }
    full = dict(kv)
    full["capture_method"] = "full_forward"
    results = compare_captures(full, kv)
    assert any(r.id == "KV3_shared_identical" and not r.passed for r in results)


def test_kv5_d0_positions_no_error():
    pos = extract_positions(5, 0)
    assert pos["shared"] >= 0
    assert pos["divergent"] == 5


def test_kv_s1_sabotage_and_unit_gate(tmp_path: Path):
    sab_ok = False
    try:
        structural_prefix_guard([1, 2], [1, 3], d=2)
    except ValueError:
        sab_ok = True
    assert sab_ok
    checks = [
        {"id": "KV1", "pass": True, "detail": "prefix guard"},
        {"id": "KV2", "pass": True, "detail": "positions"},
        {"id": "KV3", "pass": True, "detail": "shared identical"},
        {"id": "KV4", "pass": True, "detail": "layers"},
        {"id": "KV5", "pass": True, "detail": "d=0"},
        {"id": "KV6", "pass": True, "detail": "capture_method"},
        {"id": "KV7", "pass": True, "detail": "databus positions"},
    ]
    payload = write_gate(
        tmp_path / "kv_equivalence_unit_gate.json",
        gate="kv_equivalence_unit",
        pass_=True,
        checks=checks,
        sabotage=[{"id": "KV-S1", "pass": True, "detail": "prefix mismatch raised"}],
        environment={"tier": "unit", "model": None},
    )
    assert payload["pass"]

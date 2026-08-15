"""Unit tests for CaptureManifest provenance (PR1–PR8 + sabotage)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.capture_manifest import (  # noqa: E402
    CaptureManifest,
    assert_not_practice_path,
    tag_practice_pair,
    verify_on_disk,
    write_admission_gate,
    write_manifest,
)
from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    aggregate_steering_vectors,
    discover_natural_captures,
)
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402


def test_practice_never_pool_eligible(tmp_path: Path):
    practice = tmp_path / "practice_pair"
    practice.mkdir()
    (practice / "activations_at_d.pt").write_bytes(b"fake")
    (practice / "matched_prefix_spec.json").write_text(
        json.dumps({"d": 3}), encoding="utf-8"
    )
    m = tag_practice_pair(practice, pair_slug="p", d=3, prompt_len=10)
    assert m.source == "practice"
    assert m.pool_eligible(capture_dir=practice) is False
    adm = json.loads((practice / "admission_gate.json").read_text(encoding="utf-8"))
    assert adm["pool_eligible"] is False
    assert adm["source"] == "practice"


def test_natural_requires_all_gates(tmp_path: Path):
    m = CaptureManifest(
        source="natural",
        pair_slug="n1",
        d=5,
        prompt_len=10,
        calm_purity_pass=True,
        bad_purity_pass=True,
        pair_gate_pass=True,
        kv_live_gate_pass=False,
    )
    assert m.pool_eligible() is False  # PR8
    m.kv_live_gate_pass = True
    assert m.pool_eligible() is True
    m.calm_purity_pass = False
    assert m.pool_eligible() is False  # PR2


def test_verify_on_disk_tamper(tmp_path: Path):
    d = tmp_path / "pairs" / "natural" / "x"
    d.mkdir(parents=True)
    act = d / "activations_at_d.pt"
    act.write_bytes(b"abc")
    (d / "matched_prefix_spec.json").write_text("{}", encoding="utf-8")
    m = CaptureManifest(
        source="natural",
        pair_slug="x",
        d=1,
        prompt_len=1,
        calm_purity_pass=True,
        bad_purity_pass=True,
        pair_gate_pass=True,
        kv_live_gate_pass=True,
    )
    write_admission_gate(
        d, source="natural", pass_=True, checks=[], pool_eligible=True
    )
    write_manifest(m, d)
    ok = verify_on_disk(d)
    assert ok["pass"]
    # PR-S1 tamper
    act.write_bytes(b"TAMPERED")
    bad = verify_on_disk(d)
    assert bad["pass"] is False


def test_natural_pool_eligible_requires_hash_match(tmp_path: Path):
    d = tmp_path / "pairs" / "natural" / "x"
    d.mkdir(parents=True)
    act = d / "activations_at_d.pt"
    act.write_bytes(b"abc")
    (d / "matched_prefix_spec.json").write_text("{}", encoding="utf-8")
    m = CaptureManifest(
        source="natural",
        pair_slug="x",
        d=1,
        prompt_len=1,
        calm_purity_pass=True,
        bad_purity_pass=True,
        pair_gate_pass=True,
        kv_live_gate_pass=True,
    )
    write_manifest(m, d)
    loaded = CaptureManifest.from_dict(
        json.loads((d / "capture_manifest.json").read_text(encoding="utf-8"))
    )
    assert loaded.pool_eligible(capture_dir=d) is True
    act.write_bytes(b"TAMPERED")
    assert loaded.pool_eligible(capture_dir=d) is False


def test_discover_natural_empty(tmp_path: Path):
    root = tmp_path / "UNI_CAA01"
    root.mkdir()
    assert discover_natural_captures(root) == []


def test_aggregate_rejects_practice_path():
    try:
        assert_not_practice_path(Path("artifacts/caa/UNI_CAA01/practice_pair/x"))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_aggregate_rejects_practice_source():
    cap = {
        "source": "practice",
        "layers": [10, 16, 22],
        "d": 1,
        "prompt_len": 2,
        "calm": {16: {"divergent": [1.0, 0.0]}},
        "bad": {16: {"divergent": [0.0, 1.0]}},
    }
    try:
        aggregate_steering_vectors([cap], require_natural=True)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_provenance_gate_artifact(tmp_path: Path):
    gates = tmp_path / "gates"
    checks = [
        {"id": "PR1", "pass": True, "detail": "practice never pool"},
        {"id": "PR2", "pass": True, "detail": "gates required"},
        {"id": "PR3", "pass": True, "detail": "hash verify"},
        {"id": "PR4", "pass": True, "detail": "empty discover"},
        {"id": "PR5", "pass": True, "detail": "practice path raise"},
        {"id": "PR6", "pass": True, "detail": "layers"},
        {"id": "PR7", "pass": True, "detail": "divergent"},
        {"id": "PR8", "pass": True, "detail": "kv_live required"},
    ]
    sab = [
        {"id": "PR-S1", "pass": True, "detail": "tamper caught"},
        {"id": "PR-S2", "pass": True, "detail": "practice path guard"},
    ]
    payload = write_gate(
        gates / "caa_provenance_schema_gate.json",
        gate="caa_provenance_schema",
        pass_=True,
        checks=checks,
        sabotage=sab,
    )
    assert payload["pass"] is True

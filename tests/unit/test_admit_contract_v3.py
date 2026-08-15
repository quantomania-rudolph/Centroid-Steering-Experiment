"""v3 admit contract meta requirements."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_orchestrator import PIPELINE_VERSION  # noqa: E402

REQUIRED_GATES = {
    "RUN_demo_oracle",
    "RUN_demo_stress",
    "RUN_quality_suite",
    "SEM_invariants",
    "SEM_no_hardcode",
    "NARR_no_bridge_catalog",
    "NARR_no_cross_task_phrases",
    "NARR_tail_stub_distress_zero",
    "NARR_continuation_grammar",
    "NARR_balanced_backticks",
    "FMT_aider_path_outside_fence",
    "FMT_no_path_comment_in_fence",
    "PAIR_validate",
}


def test_pipeline_version_constant():
    assert PIPELINE_VERSION >= 5


def test_admitted_meta_contract_shape():
    calm_dir = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "calm_tails"
    metas = list(calm_dir.glob("*.meta.json"))
    if not metas:
        pytest.skip("no admitted tails on disk")
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "admitted":
            continue
        pv = int(meta.get("pipeline_version") or 0)
        if pv < 5:
            continue
        assert pv >= 5
        assert meta.get("repaired") is False
        assert meta.get("bridge_used") is False
        assert meta.get("sanitized") is False
        qa = meta.get("qa") or {}
        assert qa.get("llm_skipped") is not True
        gates = meta.get("gates") or {}
        for gid in REQUIRED_GATES:
            assert gid in gates, f"missing gate {gid} in {meta_path}"
            assert gates[gid] is True

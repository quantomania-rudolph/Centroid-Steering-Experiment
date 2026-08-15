"""Registry guard + clean sweep integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.calm_tail_registry import find_calm_tail, list_calm_tails  # noqa: E402


def test_v2_meta_not_visible_to_registry():
    calm_dir = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "calm_tails"
    legacy = calm_dir / "unicaa01_p06_panic__chatcmpl_44fb68e7b5cd__d3.txt"
    if not legacy.is_file():
        pytest.skip("legacy tail already swept")
    meta = legacy.with_suffix(".meta.json")
    if meta.is_file():
        data = json.loads(meta.read_text(encoding="utf-8"))
        if int(data.get("pipeline_version") or 0) < 3:
            found = find_calm_tail(
                "unicaa01_p06_panic",
                "chatcmpl_44fb68e7b5cd",
                3,
                root=calm_dir,
            )
            assert found is None


def test_list_calm_tails_v3_only():
    tails = list_calm_tails()
    for p in tails:
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert int(meta.get("pipeline_version") or 0) >= 3
        assert meta.get("status") == "admitted"

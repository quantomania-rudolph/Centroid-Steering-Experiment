"""NARR_* narrative purity gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import BRIDGE_CATALOG  # noqa: E402
from validate_narrative_purity import validate_narrative_purity  # noqa: E402


def test_bridge_catalog_detected():
    prefix = "The stub `"
    tail = BRIDGE_CATALOG[0] + "\ncode follows"
    results = validate_narrative_purity(full_text=prefix + tail, prefix_text=prefix, pair_id="p06_json_parse")
    bridge = next(r for r in results if r.gate_id == "NARR_no_bridge_catalog")
    assert bridge.pass_ is False


def test_clean_narrative_passes():
    prefix = "Fixing `"
    tail = "broken parser`.\nunicaa01_p06/main.py\n```python\ndef demo():\n    return [1,2]\n```"
    results = validate_narrative_purity(full_text=prefix + tail, prefix_text=prefix, pair_id="p06_json_parse")
    assert all(r.pass_ for r in results)


@pytest.mark.parametrize("phrase", BRIDGE_CATALOG)
def test_all_bridge_catalog_entries(phrase: str):
    prefix = "x"
    results = validate_narrative_purity(full_text=prefix + phrase, prefix_text=prefix, pair_id="p06_json_parse")
    bridge = next(r for r in results if r.gate_id == "NARR_no_bridge_catalog")
    assert bridge.pass_ is False

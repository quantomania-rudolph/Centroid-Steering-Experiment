"""Curator prompt constraint + poison scrubbing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import BRIDGE_CATALOG  # noqa: E402
from unicaa01_oracle_invariants import (  # noqa: E402
    curator_prompt_constraints,
    sanitize_bad_tail_preview,
)


def test_p06_forbids_json_in_constraints():
    text = curator_prompt_constraints("p06_json_parse")
    assert "json" in text.lower()
    assert "DO NOT" in text


def test_p06_bad_tail_scrubbed():
    raw = "import json\n```python\nreturn json.loads(s)\n```"
    out = sanitize_bad_tail_preview("p06_json_parse", raw)
    assert "import json" not in out.lower() or "FORBIDDEN" in out
    assert "```" not in out
    assert "REJECTED" in out


def test_all_pairs_scrub_code_fences():
    poison = (
        "thread-safe implementation with locks and heap pop logic.\n"
        "```python\nimport heapq\ndef demo():\n    return 1\n```\n"
        "pip install foo"
    )
    for pair_id in ("p06_json_parse", "p18_trie_prefix", "p12_fibonacci", "p01_priority_queue"):
        out = sanitize_bad_tail_preview(pair_id, poison)
        assert "```" not in out
        assert "pip install" not in out.lower()
        for bridge in BRIDGE_CATALOG:
            assert bridge.lower() not in out.lower()


def test_flash_is_default_generation():
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "curator_models.yaml").read_text(encoding="utf-8"))
    ids = [r["id"] for r in cfg["generation_models"]]
    assert ids[0] == "deepseek/deepseek-v4-flash"
    assert "openrouter/free" in ids
    assert cfg["paid_generation_fallback"]["model"] == "deepseek/deepseek-v4-flash"
    assert cfg["qa"]["model"] == "deepseek/deepseek-v4-flash"


def test_prompt_template_has_qwen_scope():
    template = (ROOT / "research" / "calm_curator_prompt.md").read_text(encoding="utf-8")
    assert "Qwen" in template or "mid-size" in template
    assert "{invariant_constraints}" in template
    assert "stdlib" in template.lower()

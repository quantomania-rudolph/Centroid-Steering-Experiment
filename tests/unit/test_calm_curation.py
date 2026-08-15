"""Unit tests for calm curation pipeline (assembler, briefs, gates)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import assemble_calm_completion, extract_main_py  # noqa: E402
from build_curator_briefs import build_all_briefs  # noqa: E402
from repair_calm_tail import repair_prefix_lock  # noqa: E402


class _FakeTok:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 1000 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i % 128) for i in ids)


def test_assemble_prefix_lock():
    tok = _FakeTok()
    prefix_ids = tok.encode("HELLO")
    assembled = assemble_calm_completion(
        prefix_token_ids=prefix_ids,
        prefix_text="HELLO",
        suffix_text=" WORLD",
        tokenizer=tok,
        d=len(prefix_ids),
        bad_token_at_d=999,
        bad_token_at_d_text="X",
    )
    assert assembled.full_token_ids[: len(prefix_ids)] == prefix_ids
    assert len(assembled.full_token_ids) > len(prefix_ids)


def test_extract_main_py():
    text = "fix\n```python\nx = 1\nprint(x)\n```\n"
    assert "x = 1" in extract_main_py(text)


def test_build_briefs_count():
    briefs = build_all_briefs()
    assert len(briefs) >= 50


def test_repair_prepends_prefix():
    tok = _FakeTok()
    prefix_ids = tok.encode("ABC")
    text, repaired = repair_prefix_lock(
        prefix_token_ids=prefix_ids,
        prefix_text="ABC",
        candidate_text="DEF",
        tokenizer=tok,
        d=len(prefix_ids),
        bad_token_at_d=999,
    )
    assert len(text) > 0
    assert repaired is True

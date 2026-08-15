"""OpenRouter client — roster generation (paid primary allowed) + QA."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from openrouter_curator_client import (  # noqa: E402
    OpenRouterCuratorClient,
    generation_models,
    load_curator_config,
)


@pytest.fixture
def cfg():
    return load_curator_config(ROOT / "config" / "curator_models.yaml")


def test_generation_roster_flash_first(cfg):
    models = generation_models(cfg)
    assert len(models) >= 1
    assert models[0].id == "deepseek/deepseek-v4-flash"
    assert models[0].tier == "paid"
    assert any(m.id == "openrouter/free" for m in models)


def test_allows_flash_generation_from_roster(cfg):
    client = OpenRouterCuratorClient(config=cfg, api_key="test-key", allow_paid=False)
    client._assert_generation_model("deepseek/deepseek-v4-flash")  # roster primary


def test_blocks_unlisted_paid_generation(cfg):
    client = OpenRouterCuratorClient(config=cfg, api_key="test-key", allow_paid=False)
    with pytest.raises(PermissionError, match="paid_model_blocked"):
        client._assert_generation_model("deepseek/deepseek-chat", last_resort=False)


def test_qa_model_is_paid(cfg):
    from openrouter_curator_client import qa_model

    qa = qa_model(cfg)
    assert qa.tier == "paid"
    assert "deepseek" in qa.id.lower()

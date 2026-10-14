"""Pytest configuration for Centroid Steering Experiment (jlens-qwen)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def qwen_bundle():
    """Load Qwen once per pytest session for slow gates."""
    from jlens_qwen.model_loader import load_qwen_coder

    bundle = load_qwen_coder(device="auto", dtype=torch.float16, prefer_local=True)
    yield bundle
    # Best-effort cleanup
    if str(bundle.device).startswith("xpu"):
        try:
            del bundle.model
            torch.xpu.empty_cache()
        except Exception:
            pass

"""QaBudgetLedger hard stop tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from openrouter_budget import BudgetExhausted, QaBudgetLedger  # noqa: E402


def test_per_brief_cap(tmp_path):
    ledger = QaBudgetLedger(max_calls_per_brief=2, max_calls_per_batch=10, ledger_path=tmp_path / "l.json")
    ledger.record_qa("b1")
    ledger.record_qa("b1")
    with pytest.raises(BudgetExhausted):
        ledger.record_qa("b1")


def test_batch_cap(tmp_path):
    ledger = QaBudgetLedger(max_calls_per_brief=5, max_calls_per_batch=2, ledger_path=tmp_path / "l.json")
    ledger.record_qa("a")
    ledger.record_qa("b")
    with pytest.raises(BudgetExhausted):
        ledger.record_qa("c")


def test_gen_fallback_cap(tmp_path):
    ledger = QaBudgetLedger(max_gen_fallback_per_batch=2, ledger_path=tmp_path / "l.json")
    ledger.record_gen_fallback()
    ledger.record_gen_fallback()
    with pytest.raises(BudgetExhausted):
        ledger.record_gen_fallback()

"""Hard call-count budget ledger for DeepSeek QA and paid generation fallback."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_PATH = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_hardening_proof" / "budget_ledger.json"


class BudgetExhausted(Exception):
    """Raised when a hard call-count cap is exceeded."""


@dataclass
class QaBudgetLedger:
    max_calls_per_brief: int = 2
    max_calls_per_batch: int = 10
    max_gen_fallback_per_batch: int = 2
    ledger_path: Path = field(default_factory=lambda: DEFAULT_LEDGER_PATH)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _qa_by_brief: dict[str, int] = field(default_factory=dict)
    _qa_batch: int = 0
    _gen_fallback_batch: int = 0

    def record_qa(self, brief_id: str) -> None:
        with self._lock:
            n = self._qa_by_brief.get(brief_id, 0) + 1
            if n > self.max_calls_per_brief:
                raise BudgetExhausted(
                    f"qa budget exhausted for brief {brief_id}: {n}>{self.max_calls_per_brief}"
                )
            if self._qa_batch + 1 > self.max_calls_per_batch:
                raise BudgetExhausted(
                    f"qa batch budget exhausted: {self._qa_batch + 1}>{self.max_calls_per_batch}"
                )
            self._qa_by_brief[brief_id] = n
            self._qa_batch += 1
            self._persist_unlocked()

    def record_gen_fallback(self) -> None:
        with self._lock:
            if self._gen_fallback_batch + 1 > self.max_gen_fallback_per_batch:
                raise BudgetExhausted(
                    f"gen fallback batch budget exhausted: "
                    f"{self._gen_fallback_batch + 1}>{self.max_gen_fallback_per_batch}"
                )
            self._gen_fallback_batch += 1
            self._persist_unlocked()

    def qa_calls_for_brief(self, brief_id: str) -> int:
        with self._lock:
            return self._qa_by_brief.get(brief_id, 0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        return {
            "qa_by_brief": dict(self._qa_by_brief),
            "qa_batch_total": self._qa_batch,
            "gen_fallback_batch_total": self._gen_fallback_batch,
            "max_calls_per_brief": self.max_calls_per_brief,
            "max_calls_per_batch": self.max_calls_per_batch,
            "max_gen_fallback_per_batch": self.max_gen_fallback_per_batch,
        }

    def _persist_unlocked(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(self._snapshot_unlocked(), indent=2), encoding="utf-8")

    @classmethod
    def from_config(cls, cfg: dict[str, Any], *, ledger_path: Path | None = None) -> QaBudgetLedger:
        qa = cfg.get("qa") or {}
        fb = cfg.get("paid_generation_fallback") or {}
        return cls(
            max_calls_per_brief=int(qa.get("max_calls_per_brief") or 2),
            max_calls_per_batch=int(qa.get("max_calls_per_batch") or 10),
            max_gen_fallback_per_batch=int(fb.get("max_calls_per_batch") or 2),
            ledger_path=ledger_path or DEFAULT_LEDGER_PATH,
        )

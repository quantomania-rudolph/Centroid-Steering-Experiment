"""Shared helpers for PrefixDivergenceTracer — tokenizer, session text, hints."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@lru_cache(maxsize=1)
def get_qwen_tokenizer() -> Any:
    """Lazy tokenizer load (no model weights)."""
    from jlens_qwen.model_loader import load_qwen_tokenizer

    return load_qwen_tokenizer()


def assistant_from_session(session_path: Path, invocation_id: str) -> str | None:
    """Full assistant text for one invocation from session JSONL."""
    if not session_path.is_file() or not invocation_id:
        return None
    for line in session_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("id") or "") != str(invocation_id):
            continue
        last_assistant: str | None = None
        for msg in row.get("messages") or []:
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = str(msg["content"])
        if last_assistant:
            return last_assistant
        # OpenAI-style completion field
        if row.get("assistant"):
            return str(row["assistant"])
    return None


def resolve_assistant_text(
    inv: dict[str, Any],
    experiment_data: dict[str, Any] | None = None,
) -> str:
    """Prefer full assistant; fall back to session JSONL, chaos token_ids, then preview."""
    if inv.get("assistant"):
        return str(inv["assistant"])
    exp = experiment_data or {}
    session_path = exp.get("session_path")
    inv_id = inv.get("id") or inv.get("invocation_id")
    completion_tokens = int(inv.get("completion_tokens") or 0)
    if session_path and inv_id:
        full = assistant_from_session(Path(session_path), str(inv_id))
        if full:
            # Session rows may only store preamble turns; reject short stale text.
            if completion_tokens <= 0 or len(full) >= max(80, completion_tokens // 4):
                return full
    chaos = inv.get("chaos") or {}
    token_ids = chaos.get("token_ids")
    if token_ids:
        tok = get_qwen_tokenizer()
        return tok.decode([int(x) for x in token_ids])
    return str(inv.get("assistant_preview") or "")


def hint_int(hints: dict[str, Any], key: str) -> int | None:
    """Read optional int from divergence_hints (0 is valid)."""
    if key not in hints:
        return None
    val = hints[key]
    if val is None:
        return None
    return int(val)

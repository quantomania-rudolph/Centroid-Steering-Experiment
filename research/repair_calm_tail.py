"""Token-safe prefix repair for calm completions (no lexical auto-strip)."""

from __future__ import annotations

import re
from typing import Any

from assemble_calm_completion import assemble_calm_completion


_NARRATIVE_REPLACEMENTS: list[tuple[str, str]] = [
    (r"NotImplementedError", "unimplemented stub"),
    (r"\bTODO\b", "pending work"),
    (r"\bFIXME\b", "known issue"),
]


def sanitize_narrative_tail(tail_text: str) -> str:
    """Remove stub/distress tokens from prose only (preserve fenced code blocks)."""
    parts = re.split(r"(```[\s\S]*?```)", tail_text)
    out: list[str] = []
    for part in parts:
        if part.startswith("```"):
            out.append(part)
            continue
        cleaned = part
        for pattern, repl in _NARRATIVE_REPLACEMENTS:
            cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
    return "".join(out)


def sanitize_full_completion(full_text: str, prefix_text: str) -> str:
    if not full_text.startswith(prefix_text):
        return full_text
    return prefix_text + sanitize_narrative_tail(full_text[len(prefix_text) :])


def repair_prefix_lock(
    *,
    prefix_token_ids: list[int],
    prefix_text: str,
    candidate_text: str,
    tokenizer: Any,
    d: int,
    bad_token_at_d: int | None = None,
    bad_token_at_d_text: str = "",
) -> tuple[str, bool]:
    """Return (repaired_text, was_repaired). Raises ValueError if still invalid."""
    assembled = assemble_calm_completion(
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        suffix_text=candidate_text,
        tokenizer=tokenizer,
        d=d,
        bad_token_at_d=bad_token_at_d,
        bad_token_at_d_text=bad_token_at_d_text,
    )
    was_repaired = not candidate_text.strip().startswith(prefix_text)
    return assembled.full_text, was_repaired

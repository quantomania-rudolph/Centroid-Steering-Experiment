"""Assemble prefix-locked calm completion from suffix-only model output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class AssembledCompletion:
    full_text: str
    full_token_ids: list[int]
    suffix_text: str
    prefix_prepended: bool
    bridge_used: bool = False


def extract_main_py(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    return ""


def tail_after_prefix(full_text: str, prefix_text: str) -> str:
    if full_text.startswith(prefix_text):
        return full_text[len(prefix_text) :]
    return full_text


def _encode(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        return [int(x) for x in ids.tolist()]
    return [int(x) for x in ids]


def _strip_leading_prefix_text(suffix: str, prefix_text: str) -> str:
    s = suffix.strip()
    if s.startswith(prefix_text):
        return s[len(prefix_text) :]
    return s


def _strip_bad_opening(suffix: str, bad_token_text: str) -> str:
    s = suffix.lstrip()
    for prefix in (
        bad_token_text,
        "NotImplementedError",
        "NotImplemented",
        "NotImplementedError()",
        "NotImplementedError.",
    ):
        if s.startswith(prefix):
            s = s[len(prefix) :].lstrip(" .,:;")
    return s


BRIDGE_CATALOG: tuple[str, ...] = (
    "thread-safe implementation with locks and heap pop logic.",
    "missing lock; implementing thread-safe run_next below.",
    "logic error; replacing stub with working implementation.",
    "unimplemented stub; providing working code below.",
)


def assemble_calm_completion(
    *,
    prefix_token_ids: list[int],
    prefix_text: str,
    suffix_text: str,
    tokenizer: Any,
    d: int,
    bad_token_at_d: int | None = None,
    bad_token_at_d_text: str = "",
    strict: bool = False,
    allow_bridge: bool = True,
) -> AssembledCompletion:
    """Character assembly + encode verification (miner file round-trip safe)."""
    prefix = list(prefix_token_ids)
    suffix = _strip_leading_prefix_text(suffix_text, prefix_text)
    if not strict:
        suffix = _strip_bad_opening(suffix, bad_token_at_d_text)

    bridges = list(BRIDGE_CATALOG)

    def _try(body: str) -> tuple[list[int], str, str] | None:
        body = body.lstrip()
        if not body:
            return None
        full_text = prefix_text + body
        ids = _encode(tokenizer, full_text)
        if ids[:d] != prefix:
            return None
        if len(ids) <= d:
            return None
        if bad_token_at_d is not None and ids[d] == bad_token_at_d:
            return None
        return ids, full_text, body

    result = _try(suffix)
    bridge_used = False
    if result is None and not strict and allow_bridge:
        for bridge in bridges:
            result = _try(bridge + " " + suffix)
            if result is not None:
                bridge_used = True
                break
        if result is None:
            result = _try(bridges[0] + " " + _strip_bad_opening(suffix, bad_token_at_d_text))
            if result is not None:
                bridge_used = True

    if result is None:
        raise ValueError(
            f"prefix lock or CD divergence failed at d={d} "
            f"(bad_token={bad_token_at_d} bad_text={bad_token_at_d_text!r})"
        )

    full_ids, full_text, suffix_body = result
    return AssembledCompletion(
        full_text=full_text,
        full_token_ids=full_ids,
        suffix_text=suffix_body,
        prefix_prepended=False,
        bridge_used=bridge_used,
    )

"""Token-class lexicon masks for B hygiene metrics (fast keyword encoding)."""

from __future__ import annotations

from typing import Any

import torch

from jlens_qwen.chaos.metrics_b import CLASS_NAMES

DISTRESS_KEYWORDS = (
    "pip",
    " install",
    "install",
    "sorry",
    "cannot",
    "can't",
    "unable",
    "error",
    "undefined",
    "NotImplemented",
    "traceback",
    "exception",
    "failed",
    "apolog",
    "Apology",
)

STUB_KEYWORDS = (
    "TODO",
    "FIXME",
    " pass",
    "pass",
    "...",
    "Ellipsis",
    "NotImplementedError",
)

PYTHON_KEYWORDS = (
    "def",
    " class",
    "class",
    "return",
    "import",
    "from",
    " if",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "try",
    "except",
    "with",
    "as",
    "True",
    "False",
    "None",
    "lambda",
    "yield",
    "async",
    "await",
    "self",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ":",
    ",",
    ".",
    "=",
    "==",
    "!=",
    "+=",
    "\n",
    "\t",
    "    ",
    "*",
    "/",
    "+",
    "-",
    "->",
)


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [int(x) for x in ids]
    except Exception:
        return []


def build_lexicon_masks(
    tokenizer: Any,
    *,
    vocab_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build BoolTensor[V] via keyword encoding (O(keywords), not O(V))."""
    V = vocab_size or int(getattr(tokenizer, "vocab_size", 0))
    if V <= 0:
        vocab = tokenizer.get_vocab()
        V = max(vocab.values()) + 1 if vocab else 1

    code = torch.zeros(V, dtype=torch.bool)
    distress = torch.zeros(V, dtype=torch.bool)
    stub = torch.zeros(V, dtype=torch.bool)
    junk = torch.zeros(V, dtype=torch.bool)

    for kw in PYTHON_KEYWORDS:
        for tid in _encode_ids(tokenizer, kw):
            if 0 <= tid < V:
                code[tid] = True
    for kw in DISTRESS_KEYWORDS:
        for tid in _encode_ids(tokenizer, kw):
            if 0 <= tid < V:
                distress[tid] = True
    for kw in STUB_KEYWORDS:
        for tid in _encode_ids(tokenizer, kw):
            if 0 <= tid < V:
                stub[tid] = True

    # Junk: a few known bad patterns if they tokenize as single tokens
    for junk_s in ("ERCHANTABILITY", "AnimationFrame", "iał", "ставил"):
        for tid in _encode_ids(tokenizer, junk_s):
            if 0 <= tid < V:
                junk[tid] = True

    return {
        "code": code,
        "distress": distress,
        "stub": stub,
        "junk": junk,
    }


def lexicon_summary(masks: dict[str, torch.Tensor]) -> dict[str, int]:
    return {name: int(masks[name].sum().item()) for name in CLASS_NAMES}

"""Prefix-locked calm-tail builder for Method B natural pairs (UNI_CAA01 § Phase 2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CalmTailSpec:
    """Calm completion that shares exact prefix_token_ids through d-1."""

    d: int
    prefix_token_ids: list[int]
    calm_full_token_ids: list[int]
    calm_tail_token_ids: list[int]
    calm_text: str
    source_path: str = ""
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def validate_prefix_lock(
    calm_full_token_ids: list[int],
    prefix_token_ids: list[int],
    d: int,
) -> dict[str, Any]:
    """Round-trip lock: calm_full[:d] must equal prefix_token_ids exactly."""
    checks: list[dict[str, Any]] = []
    prefix = list(prefix_token_ids)
    calm = list(calm_full_token_ids)
    expected_d = int(d)

    checks.append(
        {
            "id": "CT_d_positive",
            "pass": expected_d > 0,
            "detail": f"d={expected_d}",
        }
    )
    checks.append(
        {
            "id": "CT_prefix_len",
            "pass": len(prefix) == expected_d,
            "detail": f"len(prefix)={len(prefix)} d={expected_d}",
        }
    )
    checks.append(
        {
            "id": "CT_calm_long_enough",
            "pass": len(calm) > expected_d,
            "detail": f"len(calm)={len(calm)} need>{expected_d}",
        }
    )
    head = calm[:expected_d] if len(calm) >= expected_d else calm
    checks.append(
        {
            "id": "CT_roundtrip_prefix_eq",
            "pass": head == prefix,
            "detail": (
                f"lcp={longest_common_prefix(head, prefix)} "
                f"expected_d={expected_d}"
            ),
        }
    )
    # Calm must extend past the shared prefix (first post-d token is the tail start).
    checks.append(
        {
            "id": "CT_diverges_at_d",
            "pass": len(calm) > expected_d,
            "detail": (
                f"len(calm)={len(calm)} need>{expected_d}; "
                f"first_post_d={calm[expected_d] if len(calm) > expected_d else None}"
            ),
        }
    )

    ok = all(c["pass"] for c in checks)
    return {"pass": ok, "checks": checks}


def build_calm_tail_from_ids(
    *,
    prefix_token_ids: list[int],
    calm_full_token_ids: list[int],
    d: int | None = None,
    calm_text: str = "",
    source_path: str = "",
) -> CalmTailSpec:
    """Build CalmTailSpec from curated token IDs; raises if prefix lock fails."""
    prefix = list(prefix_token_ids)
    calm = list(calm_full_token_ids)
    d_use = int(d) if d is not None else len(prefix)
    gate = validate_prefix_lock(calm, prefix, d_use)
    if not gate["pass"]:
        raise ValueError(f"calm-tail prefix lock failed: {gate['checks']}")
    if len(calm) <= d_use:
        raise ValueError("calm completion must extend past d")
    return CalmTailSpec(
        d=d_use,
        prefix_token_ids=prefix,
        calm_full_token_ids=calm,
        calm_tail_token_ids=calm[d_use:],
        calm_text=calm_text,
        source_path=source_path,
        notes=["curated_file", "teacher_forced_only"],
    )


def build_calm_tail_from_file(
    *,
    prefix_token_ids: list[int],
    d: int,
    calm_completion_path: Path | str,
    tokenizer: Any,
) -> CalmTailSpec:
    """Load curated calm completion text, tokenize, enforce prefix lock.

    Free-gen is forbidden — only curated files.
    """
    path = Path(calm_completion_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if hasattr(ids, "tolist"):
            ids = [int(x) for x in ids.tolist()]
        else:
            ids = [int(x) for x in ids]
    else:
        raise TypeError("tokenizer must provide encode()")
    return build_calm_tail_from_ids(
        prefix_token_ids=prefix_token_ids,
        calm_full_token_ids=ids,
        d=d,
        calm_text=text,
        source_path=str(path),
    )


def reject_free_generate() -> None:
    """v1: free-generate calm tails are never admitted."""
    raise NotImplementedError(
        "free_generate calm tails are forbidden for Method B admission — "
        "use curated completion files only (UNI_CAA01 Phase 2)"
    )

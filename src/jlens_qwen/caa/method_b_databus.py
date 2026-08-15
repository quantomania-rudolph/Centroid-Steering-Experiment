"""Method B matched-prefix databus — canonical per-invocation record for CAA extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TailKind = Literal["calm", "bad", "observed"]


@dataclass
class MethodBRecord:
    """All fields needed to mine û_beh / û_cond at divergence (UNI_CAA01 §2.1)."""

    task_slug: str
    experiment: str
    invocation_id: str
    user_prompt: str
    # Divergence indices (completion-step space, 0-indexed).
    d: int
    prefix_end: int
    d_lexicon: int | None = None
    d_flags: int | None = None
    d_recipe: int | None = None
    # Token streams.
    prefix_token_ids: list[int] = field(default_factory=list)
    tail_token_ids: list[int] = field(default_factory=list)
    full_completion_token_ids: list[int] = field(default_factory=list)
    prefix_text: str = ""
    tail_text: str = ""
    full_completion_text: str = ""
    # Extraction positions (absolute index in prompt+completion sequence).
    extract_positions: list[int] = field(default_factory=list)
    extract_layers: list[int] = field(default_factory=lambda: [10, 16, 22])
    tail_kind: TailKind = "observed"
    # Tracer / chaos linkage.
    confidence: str = "none"
    eligible: bool = False
    divergence_class: str = "none"
    chaos_version: int | None = None
    token_ids_match: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def prompt_len_hint(self) -> int | None:
        """Set by builder when prompt token length is known."""
        return getattr(self, "_prompt_len", None)


def build_method_b_record(
    *,
    task_slug: str,
    experiment: str,
    invocation_id: str,
    user_prompt: str,
    divergence: dict[str, Any],
    tail_token_ids: list[int] | None = None,
    tail_text: str = "",
    tail_kind: TailKind = "observed",
    prompt_len: int | None = None,
    extract_layers: list[int] | None = None,
    fallback_d: int | None = None,
    fallback_prefix_token_ids: list[int] | None = None,
) -> MethodBRecord | None:
    """Build databus record from PrefixDivergenceTracer output + optional tail."""
    mb = divergence.get("method_b_record") or {}
    d = divergence.get("d_consensus")
    if d is None and mb.get("d") is not None:
        d = int(mb["d"])
    if d is None and fallback_d is not None:
        d = int(fallback_d)
    if d is None:
        return None

    # Prefer explicit fallback (extract-d remapped prefix) over tracer prefix when
    # provided — tracer prefix length tracks d_tracer, which may differ from d_extract.
    if fallback_prefix_token_ids is not None:
        prefix_ids = list(fallback_prefix_token_ids)
    else:
        prefix_ids = list(
            divergence.get("prefix_token_ids")
            or mb.get("prefix_token_ids")
            or []
        )
    if not prefix_ids and mb.get("prefix_text"):
        notes = ["prefix_token_ids missing — text-only prefix"]
    else:
        notes = list(divergence.get("notes") or [])

    prefix_text = str(divergence.get("prefix_text") or mb.get("prefix_text") or "")
    tail_ids = list(tail_token_ids or [])
    full_ids = list(prefix_ids) + list(tail_ids)

    layers = list(extract_layers or [10, 16, 22])
    extract_pos: list[int] = []
    if prompt_len is not None:
        # Absolute positions in [prompt || completion] sequence.
        if d > 0:
            extract_pos.append(prompt_len + d - 1)  # last shared
        extract_pos.append(prompt_len + d)  # first divergent

    rec = MethodBRecord(
        task_slug=task_slug,
        experiment=experiment,
        invocation_id=invocation_id,
        user_prompt=user_prompt,
        d=int(d),
        prefix_end=int(d) - 1 if int(d) > 0 else -1,
        d_lexicon=divergence.get("d_lexicon"),
        d_flags=divergence.get("d_flags"),
        d_recipe=divergence.get("d_recipe"),
        prefix_token_ids=prefix_ids,
        tail_token_ids=tail_ids,
        full_completion_token_ids=full_ids,
        prefix_text=prefix_text,
        tail_text=tail_text,
        full_completion_text=prefix_text + tail_text if prefix_text or tail_text else "",
        extract_positions=extract_pos,
        extract_layers=layers,
        tail_kind=tail_kind,
        confidence=str(divergence.get("confidence") or "none"),
        eligible=bool(divergence.get("eligible")),
        divergence_class=str(divergence.get("divergence_class") or "none"),
        chaos_version=None,
        token_ids_match=bool(divergence.get("token_ids_match")),
        notes=notes,
    )
    if prompt_len is not None:
        rec._prompt_len = prompt_len  # type: ignore[attr-defined]
    return rec


def validate_method_b_record(
    rec: MethodBRecord | dict[str, Any],
    *,
    prompt_len: int | None = None,
) -> dict[str, Any]:
    """Structural checks before activation extraction."""
    if isinstance(rec, MethodBRecord):
        r = rec
        d = r.d
        prefix_ids = r.prefix_token_ids
        tail_ids = r.tail_token_ids
        full_ids = r.full_completion_token_ids
        extract_positions = r.extract_positions
        extract_layers = r.extract_layers
    else:
        d = int(rec.get("d", -1))
        prefix_ids = list(rec.get("prefix_token_ids") or [])
        tail_ids = list(rec.get("tail_token_ids") or [])
        full_ids = list(rec.get("full_completion_token_ids") or [])
        extract_positions = list(rec.get("extract_positions") or [])
        extract_layers = list(rec.get("extract_layers") or [10, 16, 22])

    checks: list[dict[str, Any]] = []

    checks.append(
        {
            "id": "M1_d_positive",
            "pass": d >= 0,
            "detail": f"d={d}",
        }
    )
    checks.append(
        {
            "id": "M2_prefix_len",
            "pass": len(prefix_ids) == d if d > 0 else len(prefix_ids) == 0,
            "detail": f"len(prefix)={len(prefix_ids)} d={d}",
        }
    )
    if full_ids or tail_ids:
        checks.append(
            {
                "id": "M3_full_completion",
                "pass": len(full_ids) == len(prefix_ids) + len(tail_ids),
                "detail": (
                    f"full={len(full_ids)} "
                    f"prefix+tail={len(prefix_ids)+len(tail_ids)}"
                ),
            }
        )
    if prompt_len is not None and extract_positions:
        expected = []
        if d > 0:
            expected.append(prompt_len + d - 1)
        expected.append(prompt_len + d)
        checks.append(
            {
                "id": "M4_extract_positions",
                "pass": extract_positions == expected,
                "detail": f"got={extract_positions} expected={expected}",
            }
        )
    checks.append(
        {
            "id": "M5_layers",
            "pass": len(extract_layers) >= 1 and all(isinstance(L, int) for L in extract_layers),
            "detail": f"layers={extract_layers}",
        }
    )
    ok = all(c["pass"] for c in checks)
    record_out = rec.to_dict() if isinstance(rec, MethodBRecord) else dict(rec)
    return {"pass": ok, "checks": checks, "record": record_out}

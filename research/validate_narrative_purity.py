"""L4 narrative purity gates — bridge catalog, cross-task phrases, stub/distress scores."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from assemble_calm_completion import BRIDGE_CATALOG, tail_after_prefix
from caa_error_combo_study import score_text
from unicaa01_oracle_invariants import CONCURRENCY_PAIR_IDS, SCHEDULER_PAIR_IDS

_SCHEDULER_ONLY_PHRASES = (
    "run_next",
    "heap pop",
    "heapq.heappop",
)
_LOCK_PHRASES = (
    "thread-safe",
    "threading.Lock",
)
_CROSS_TASK_PHRASES = _SCHEDULER_ONLY_PHRASES + _LOCK_PHRASES

# Prompt-paste imperatives anywhere in the first narrative block.
# Case-insensitive for strong verbs; keep Add capital-only to avoid "add the values".
_IMPERATIVE_RE = re.compile(
    r"\b(?:Implement|Write|Create|Edit|Build)\b|Fix the|\bAdd\b",
    re.IGNORECASE,
)
# Re-tighten Add to capital-only after IGNORECASE match filtering in validate.

_PATH_LINE = re.compile(r"(?:^|[\s`])[\w./\\-]+/main\.py\s*$")


@dataclass
class NarrativeGateResult:
    gate_id: str
    pass_: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.gate_id, "pass": self.pass_, "detail": self.detail}


def _first_narrative_block(tail: str) -> str:
    """Prose after the locked prefix, before path line or code fence."""
    lines: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            break
        if stripped.endswith("/main.py") or stripped.endswith("\\main.py"):
            break
        if _PATH_LINE.search(stripped):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def validate_narrative_purity(
    *,
    full_text: str,
    prefix_text: str,
    pair_id: str,
) -> list[NarrativeGateResult]:
    """Run NARR_* gates on tail narrative after prefix lock."""
    tail = tail_after_prefix(full_text, prefix_text)
    results: list[NarrativeGateResult] = []

    hits = [b for b in BRIDGE_CATALOG if b.lower() in tail.lower()]
    results.append(
        NarrativeGateResult(
            "NARR_no_bridge_catalog",
            len(hits) == 0,
            f"hits={hits}" if hits else "clean",
        )
    )

    cross_hits: list[str] = []
    if pair_id not in SCHEDULER_PAIR_IDS:
        for phrase in _SCHEDULER_ONLY_PHRASES:
            if re.search(re.escape(phrase), tail, re.IGNORECASE):
                cross_hits.append(phrase)
    if pair_id not in CONCURRENCY_PAIR_IDS:
        for phrase in _LOCK_PHRASES:
            if re.search(re.escape(phrase), tail, re.IGNORECASE):
                cross_hits.append(phrase)
    results.append(
        NarrativeGateResult(
            "NARR_no_cross_task_phrases",
            len(cross_hits) == 0,
            f"hits={cross_hits}" if cross_hits else "clean",
        )
    )

    scores = score_text(tail)
    stub = int(scores.get("stub_score") or 0)
    distress = int(scores.get("distress_score") or 0)
    results.append(
        NarrativeGateResult(
            "NARR_tail_stub_distress_zero",
            stub == 0 and distress == 0,
            f"stub={stub} distress={distress}",
        )
    )

    narrative = _first_narrative_block(tail)
    imperative_hits: list[str] = []
    for m in _IMPERATIVE_RE.finditer(narrative):
        hit = m.group(0)
        # Allow lowercase/common "add" but reject prompt-paste "Add".
        if hit.lower() == "add" and hit != "Add":
            continue
        imperative_hits.append(hit)
    imperative_hits = sorted(set(imperative_hits))
    results.append(
        NarrativeGateResult(
            "NARR_continuation_grammar",
            len(imperative_hits) == 0,
            f"prompt_paste={imperative_hits}" if imperative_hits else "continuation ok",
        )
    )

    # Include locked prefix so an opening tick in the prefix must be closed in prose.
    prose_span = f"{prefix_text}{narrative}"
    tick_count = prose_span.count("`")
    results.append(
        NarrativeGateResult(
            "NARR_balanced_backticks",
            tick_count % 2 == 0,
            f"backtick_count={tick_count} narrative={narrative[:80]!r}",
        )
    )
    return results


def narrative_all_pass(results: list[NarrativeGateResult]) -> bool:
    return all(r.pass_ for r in results)

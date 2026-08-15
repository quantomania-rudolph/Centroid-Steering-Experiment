"""Per-pair invariant and stress specs for UNI_CAA01 demo oracle v3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from assemble_calm_completion import BRIDGE_CATALOG
from unicaa01_registry import PAIRS_ALL

InvariantKind = Literal["require_symbol", "forbid_import", "forbid_literal_return"]


@dataclass(frozen=True)
class InvariantSpec:
    kind: InvariantKind
    value: str
    detail: str = ""


@dataclass(frozen=True)
class StressSpec:
    threads: int = 10
    wall_sec: float = 15.0


# Tier B/C specs — tier A (demo exists, no pass-only stub) is always enforced in check_invariants.
_PAIR_INVARIANTS: dict[str, tuple[InvariantSpec, ...]] = {
    "p01_priority_queue": (InvariantSpec("require_symbol", "threading.Lock", "scheduler lock"),),
    "p06_json_parse": (InvariantSpec("forbid_import", "json", "minimal JSON parser — no stdlib json"),),
    "p08_rate_limiter": (InvariantSpec("require_symbol", "threading", "rate limiter concurrency"),),
    "p09_thread_pool": (InvariantSpec("require_symbol", "executor", "thread pool executor"),),
    "p12_fibonacci": (InvariantSpec("forbid_literal_return", "55", "no bare return 55 in demo"),),
    "p42_semaphore": (InvariantSpec("require_symbol", "threading", "semaphore concurrency"),),
    "p43_rwlock": (InvariantSpec("require_symbol", "threading", "rwlock concurrency"),),
    "p44_barrier": (InvariantSpec("require_symbol", "threading", "barrier concurrency"),),
    "p45_producer_consumer": (InvariantSpec("require_symbol", "threading", "producer-consumer concurrency"),),
}

_STRESS_PAIRS: dict[str, StressSpec] = {
    "p01_priority_queue": StressSpec(),
    "p08_rate_limiter": StressSpec(),
    "p09_thread_pool": StressSpec(),
    "p42_semaphore": StressSpec(),
    "p43_rwlock": StressSpec(),
    "p44_barrier": StressSpec(),
    "p45_producer_consumer": StressSpec(),
}

# Scheduler-like pairs for cross-task phrase detection (non-scheduler must not echo these).
SCHEDULER_PAIR_IDS = frozenset({"p01_priority_queue"})

# Pairs that legitimately discuss locks / thread-safety in narrative + code.
CONCURRENCY_PAIR_IDS = frozenset(
    {
        "p01_priority_queue",
        "p08_rate_limiter",
        "p09_thread_pool",
        "p42_semaphore",
        "p43_rwlock",
        "p44_barrier",
        "p45_producer_consumer",
    }
)
# Patterns that poison generation when copied from panic tails (all pairs).
_GLOBAL_POISON_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"```[\s\S]*?```", re.MULTILINE),
    re.compile(r"(?im)^\s*import\s+\w+.*$", re.MULTILINE),
    re.compile(r"(?im)^\s*from\s+\w+.*import.*$", re.MULTILINE),
    re.compile(r"(?im)^\s*def\s+\w+\s*\(.*$", re.MULTILINE),
    re.compile(r"(?im)^\s*class\s+\w+.*$", re.MULTILINE),
    re.compile(r"(?im)^\s*return\s+.*$", re.MULTILINE),
    re.compile(r"(?i)NotImplementedError"),
    re.compile(r"(?i)\bpip install\b"),
)


def invariants_for(pair_id: str) -> tuple[InvariantSpec, ...]:
    return _PAIR_INVARIANTS.get(pair_id, ())


def stress_for(pair_id: str) -> StressSpec | None:
    return _STRESS_PAIRS.get(pair_id)


def curator_prompt_constraints(pair_id: str) -> str:
    """Human-readable invariant lines injected into generation/QA prompts."""
    lines: list[str] = []
    for spec in invariants_for(pair_id):
        if spec.kind == "forbid_import":
            lines.append(
                f"DO NOT import or call stdlib `{spec.value}` anywhere in main.py — {spec.detail}."
            )
        elif spec.kind == "require_symbol":
            lines.append(f"main.py MUST use `{spec.value}` — {spec.detail}.")
        elif spec.kind == "forbid_literal_return":
            lines.append(
                f"demo() MUST NOT be a single literal return of {spec.value} — {spec.detail}."
            )
    if stress_for(pair_id) is not None:
        lines.append("Implementation must be safe under concurrent demo() stress (thread-safe).")
    if pair_id not in SCHEDULER_PAIR_IDS:
        lines.append(
            "Do NOT use scheduler boilerplate "
            "(heap pop, run_next) — wrong task."
        )
        if pair_id not in CONCURRENCY_PAIR_IDS:
            lines.append(
                "Do NOT use concurrency/lock boilerplate "
                "(thread-safe locks, threading.Lock) — wrong task."
            )
    if not lines:
        return "- Follow success criteria; implement real logic (no stubs)."
    return "\n".join(f"- {line}" for line in lines)


def sanitize_bad_tail_preview(pair_id: str, preview: str) -> str:
    """Strip copyable poison from panic-tail previews for ALL pairs.

    Models copy bad-tail code/imports/bridge phrases. We keep a short prose
    warning only — never fenced code, imports, or BRIDGE_CATALOG strings.
    """
    text = preview or ""
    for pat in _GLOBAL_POISON_PATTERNS:
        text = pat.sub(" ", text)
    # Orphan fence markers (unclosed ```) and inline backticks
    text = text.replace("```", " ")
    text = text.replace("`", "'")
    for bridge in BRIDGE_CATALOG:
        text = re.sub(re.escape(bridge), "[FORBIDDEN bridge phrase]", text, flags=re.I)
    for phrase in ("thread-safe", "run_next", "heap pop", "heapq.heappop", "json.loads"):
        if phrase == "json.loads":
            text = re.sub(re.escape(phrase), f"[scrubbed:{phrase}]", text, flags=re.I)
            continue
        if phrase in ("thread-safe",) and pair_id in CONCURRENCY_PAIR_IDS:
            continue
        if phrase in ("run_next", "heap pop", "heapq.heappop") and pair_id in SCHEDULER_PAIR_IDS:
            continue
        if pair_id in SCHEDULER_PAIR_IDS and phrase == "thread-safe":
            continue
        text = re.sub(re.escape(phrase), f"[scrubbed:{phrase}]", text, flags=re.I)

    for spec in invariants_for(pair_id):
        if spec.kind == "forbid_import":
            text = re.sub(
                rf"\b{re.escape(spec.value)}\b",
                f"[FORBIDDEN import:{spec.value}]",
                text,
                flags=re.I,
            )

    # Drop aider-style path lines that invite copying the listing format from panic
    text = re.sub(r"(?i)\b\S+/main\.py\b", "[task main.py]", text)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 220:
        text = text[:220] + "…"

    header = (
        "REJECTED panic-tail summary (prose only — DO NOT copy code/imports/phrases from it). "
        "Write a clean task-specific fix instead."
    )
    if not text:
        return header + " [empty after scrub]"
    return f"{header} {text}"


def all_invariant_pair_ids() -> list[str]:
    return sorted(p.pair_id for p in PAIRS_ALL)


def verify_invariant_coverage() -> dict[str, object]:
    registry_ids = sorted(p.pair_id for p in PAIRS_ALL)
    from unicaa01_demo_oracles import ORACLES

    missing = [pid for pid in registry_ids if pid not in ORACLES]
    return {
        "n_pairs": len(registry_ids),
        "n_with_extra_invariants": len(_PAIR_INVARIANTS),
        "n_stress": len(_STRESS_PAIRS),
        "missing_oracles": missing,
        "pass": len(missing) == 0 and len(registry_ids) == 66,
    }

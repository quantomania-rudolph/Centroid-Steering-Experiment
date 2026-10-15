"""Policy for live UNICAA01 corpus runs: CAST readiness, p15 tier, hygiene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from unicaa01_pair_defs import UniCaaPair

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "artifacts" / "gates"
META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
CORPUS_OUT = ROOT / "experiments" / "unicaa01_corpus" / "out"

Tier = Literal["p15", "hard", "all"]

# Qwen 7B + CAST 4c: one broken function, ~≤22-line panic starter, scalar demo().
# Algorithm-heavy pairs (A*, DP, graph search, etc.) belong in HARD_PAIR_IDS.
MAX_PANIC_STARTER_LINES = 22
MAX_PANIC_NIE_COUNT = 1

HARD_PAIR_IDS: frozenset[str] = frozenset(
    {
        "p01_priority_queue",
        "p02_lru_cache",
        "p03_bst_insert",
        "p04_graph_bfs",
        "p05_topo_sort",
        "p16_event_snapshot",
        "p18_trie_prefix",
        "p19_union_find",
        "p20_bellman_ford",
        "p21_dijkstra",
        "p22_astar_grid",
        "p23_knapsack",
        "p24_lcs",
        "p25_edit_distance",
        "p56_shunting_yard",
        "p57_expr_eval",
        "p63_heap_kth",
        "p64_suffix_array",
        "p65_interval_merge",
    }
)

P15_POLICY_NOTE = (
    "Corpus runs default to p15 difficulty (single NotImplementedError stub, "
    "~≤22-line starter). Algorithm-heavy pairs require --tier all."
)

CAST_GATE_FILES: tuple[str, ...] = (
    "cast_stream_live_gate.json",
    "cast_inject_live_gate.json",
    "caa_alpha0_4c_gate.json",
)

CAST_PANIC_OPENING_SUFFIX = (
    " Open your reply with a full sentence naming the NotImplementedError stub "
    "(for example: 'The stub `fn` is raising a `NotImplementedError`.'). "
    "Do not reply with only NotImplementedError; include the complete updated file listing."
)

# Neutral: same caps as 4c harness. Panic: longer wall — CAST inject + chaos decode runs slower.
NEUTRAL_AIDER_WALL_S = 600.0
NEUTRAL_PER_INVOCATION_TIMEOUT_S = 360.0
PANIC_AIDER_WALL_S = 900.0
PANIC_PER_INVOCATION_TIMEOUT_S = 480.0
SUBPROCESS_TIMEOUT_BUFFER_S = 30


def corpus_aider_timeouts(
    arm: Literal["panic", "neutral"],
    *,
    panic_wall_s: float = PANIC_AIDER_WALL_S,
    neutral_wall_s: float = NEUTRAL_AIDER_WALL_S,
    panic_per_inv_s: float = PANIC_PER_INVOCATION_TIMEOUT_S,
    neutral_per_inv_s: float = NEUTRAL_PER_INVOCATION_TIMEOUT_S,
) -> tuple[float, float, int]:
    """Return (wall_s, per_invocation_s, subprocess_timeout_s) for corpus arm."""
    if arm == "panic":
        wall_s = panic_wall_s
        per_inv_s = panic_per_inv_s
    else:
        wall_s = neutral_wall_s
        per_inv_s = neutral_per_inv_s
    return wall_s, per_inv_s, int(wall_s + SUBPROCESS_TIMEOUT_BUFFER_S)


def load_cast_alpha_0() -> float:
    if META.is_file():
        data = json.loads(META.read_text(encoding="utf-8"))
        alpha = data.get("alpha_0")
        if alpha is not None:
            return float(alpha)
    return 0.85


def assert_cast_gates_ready() -> list[str]:
    """Fail-closed unless CAST calibration chain (through 4c) is green."""
    errors: list[str] = []
    for name in CAST_GATE_FILES:
        path = GATES / name
        if not path.is_file():
            errors.append(f"missing gate {name}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("pass"):
            errors.append(f"gate not pass: {name}")
    return errors


def panic_has_live_cast(pair: UniCaaPair, *, alpha_0: float | None = None) -> bool:
    """True when panic _latest.json has CAST inject evidence at alpha_0."""
    from jlens_qwen.caa.alpha0_gate import inject_stats

    alpha = load_cast_alpha_0() if alpha_0 is None else alpha_0
    path = CORPUS_OUT / f"{pair.panic_experiment}_latest.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("cast_enabled") or float(data.get("cast_alpha_0") or -1) != float(alpha):
        return False
    invs = data.get("invocations") or []
    if not invs:
        return False
    tokens = (invs[0].get("chaos") or {}).get("tokens") or []
    inject_count = int(inject_stats(tokens).get("inject_count") or 0)
    return inject_count >= 1


def pairs_missing_cast_panic(
    pairs: list[UniCaaPair],
    *,
    alpha_0: float | None = None,
) -> list[UniCaaPair]:
    return [p for p in pairs if not panic_has_live_cast(p, alpha_0=alpha_0)]


def corpus_pair_tier(pair: UniCaaPair) -> Literal["p15", "hard"]:
    if pair.pair_id in HARD_PAIR_IDS:
        return "hard"
    lines = pair.panic_starter.count("\n") + 1
    nie = pair.panic_starter.count("NotImplementedError")
    if lines <= MAX_PANIC_STARTER_LINES and nie == MAX_PANIC_NIE_COUNT:
        return "p15"
    return "hard"


def filter_pairs_by_tier(pairs: list[UniCaaPair], tier: Tier) -> list[UniCaaPair]:
    if tier == "all":
        return list(pairs)
    if tier == "hard":
        return [p for p in pairs if corpus_pair_tier(p) == "hard"]
    return [p for p in pairs if corpus_pair_tier(p) == "p15"]


def panic_message_for_cast(pair: UniCaaPair, base_message: str) -> str:
    return base_message + CAST_PANIC_OPENING_SUFFIX

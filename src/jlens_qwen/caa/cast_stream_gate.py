"""CAST live stream gate checks (token-list scoring, no HTTP/Aider)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

CAST_EVAL_P95_MAX_US = 60_000.0
MIN_PANIC_TOKENS = 32
STUB_DISTRESS_RECIPE = "B_STUB_DISTRESS"
C_PLUS_G_MINUS_MAX_RATE = 0.15
PANIC_GAMMA_MEDIAN_MIN = 0.5
CLEAN_G_STREAM_RATE_MAX = 0.05
CLEAN_GAMMA_MEDIAN_MAX = 0.1

CheckResult = dict[str, Any]


def chaos_tokens_from_session(session: Path | dict[str, Any]) -> list[dict[str, Any]]:
    """Return chaos tokens from the invocation with the longest token stream."""
    if isinstance(session, Path):
        data = json.loads(session.read_text(encoding="utf-8"))
    else:
        data = session
    invs = data.get("invocations") or []
    if not invs:
        return []
    best = max(
        invs,
        key=lambda inv: len((inv.get("chaos") or {}).get("tokens") or []),
    )
    chaos = best.get("chaos") or {}
    return list(chaos.get("tokens") or [])


def cast_stats(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    casts = [t.get("cast") or {} for t in tokens if t.get("cast")]
    if not casts:
        return {"n": 0}
    g_streams = [int(c.get("g_stream") or 0) for c in casts]
    gammas = [float(c.get("gamma") or 0) for c in casts]
    injects = [bool(c.get("inject_applied")) for c in casts]
    eval_us = [float(c.get("cast_eval_us") or 0) for c in casts]
    return {
        "n": len(casts),
        "g_stream_rate": sum(g_streams) / len(g_streams),
        "gamma_median": statistics.median(gammas) if gammas else 0.0,
        "inject_any": any(injects),
        "cast_eval_p95_us": sorted(eval_us)[int(len(eval_us) * 0.95)] if eval_us else 0.0,
    }


def _token_cast(token: dict[str, Any]) -> dict[str, Any]:
    return token.get("cast") or {}


def _g_stream_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in tokens if int(_token_cast(t).get("g_stream") or 0) == 1]


def _n_g_stream(tokens: list[dict[str, Any]]) -> int:
    return len(_g_stream_tokens(tokens))


def _has_cast_shadow(token: dict[str, Any]) -> bool:
    cast = token.get("cast")
    if not isinstance(cast, dict):
        return False
    return "g_stream" in cast and "gamma" in cast


def _recipe_id(token: dict[str, Any]) -> str | None:
    cast = _token_cast(token)
    rid = cast.get("recipe_id") or token.get("recipe_id")
    return str(rid) if rid else None


def check_a1(clean_stats: dict[str, Any]) -> CheckResult:
    return {
        "id": "A1",
        "pass": clean_stats.get("n", 0) > 0
        and clean_stats.get("g_stream_rate", 1) <= CLEAN_G_STREAM_RATE_MAX,
        "detail": str(clean_stats),
    }


def check_a2(clean_stats: dict[str, Any]) -> CheckResult:
    return {
        "id": "A2",
        "pass": clean_stats.get("gamma_median", 1) < CLEAN_GAMMA_MEDIAN_MAX,
        "detail": f"gamma_median={clean_stats.get('gamma_median')}",
    }


def check_a3(clean_stats: dict[str, Any]) -> CheckResult:
    return {
        "id": "A3",
        "pass": not clean_stats.get("inject_any"),
        "detail": "no inject_applied on clean",
    }


def check_b1(panic_tokens: list[dict[str, Any]]) -> CheckResult:
    n = len(panic_tokens)
    casts = [_token_cast(t) for t in panic_tokens]
    recipe_hits = sum(1 for c in casts if c.get("g_recipe") == 1)
    has_stub_distress = any(_recipe_id(t) == STUB_DISTRESS_RECIPE for t in panic_tokens)
    recipe_ok = recipe_hits >= 1 or has_stub_distress
    n_ok = n >= MIN_PANIC_TOKENS
    return {
        "id": "B1",
        "pass": recipe_ok and n_ok,
        "detail": (
            f"n={n} min={MIN_PANIC_TOKENS} recipe_hits={recipe_hits} "
            f"stub_distress={has_stub_distress}"
        ),
    }


def check_b2(panic_stats: dict[str, Any]) -> CheckResult:
    return {
        "id": "B2",
        "pass": panic_stats.get("g_stream_rate", 0) > 0,
        "detail": f"g_stream_rate={panic_stats.get('g_stream_rate')}",
    }


def check_b3(panic_tokens: list[dict[str, Any]]) -> CheckResult:
    n_g = _n_g_stream(panic_tokens)
    panic_window_gammas = [
        float(_token_cast(t).get("gamma") or 0) for t in _g_stream_tokens(panic_tokens)
    ]
    if n_g == 0:
        passed = False
        median_gamma: float | str = "n/a"
    else:
        median_gamma = statistics.median(panic_window_gammas)
        passed = median_gamma >= PANIC_GAMMA_MEDIAN_MIN
    return {
        "id": "B3",
        "pass": passed,
        "detail": f"n_g_stream={n_g} panic_gamma_median={median_gamma}",
    }


def check_b4(panic_tokens: list[dict[str, Any]]) -> CheckResult:
    g_stream_tokens = _g_stream_tokens(panic_tokens)
    n_g = len(g_stream_tokens)
    if n_g == 0:
        return {
            "id": "B4",
            "pass": False,
            "detail": "n_g_stream=0 c_plus_g_minus_rate=n/a",
        }
    low_gamma_on_stream = sum(
        1 for t in g_stream_tokens if float(_token_cast(t).get("gamma") or 0) < PANIC_GAMMA_MEDIAN_MIN
    )
    c_plus_g_minus_rate = low_gamma_on_stream / n_g
    return {
        "id": "B4",
        "pass": c_plus_g_minus_rate <= C_PLUS_G_MINUS_MAX_RATE,
        "detail": f"n_g_stream={n_g} c_plus_g_minus_rate={c_plus_g_minus_rate:.4f}",
    }


def check_b5(panic_tokens: list[dict[str, Any]]) -> CheckResult:
    bad_and = 0
    for token in panic_tokens:
        cast = _token_cast(token)
        alpha = float(cast.get("alpha_eff") or 0)
        if alpha <= 0:
            continue
        if not (
            int(cast.get("g_stream") or 0) == 1
            and int(cast.get("g_recipe") or 0) == 1
            and float(cast.get("gamma") or 0) > 0
        ):
            bad_and += 1
    return {
        "id": "B5",
        "pass": bad_and == 0,
        "detail": f"alpha_without_and_gates={bad_and}",
    }


def check_l1(panic_stats: dict[str, Any]) -> CheckResult:
    return {
        "id": "L1",
        "pass": panic_stats.get("cast_eval_p95_us", 1e9) <= CAST_EVAL_P95_MAX_US,
        "detail": f"p95_us={panic_stats.get('cast_eval_p95_us')}",
    }


def check_shadow(panic_tokens: list[dict[str, Any]]) -> CheckResult:
    if not panic_tokens:
        return {
            "id": "SHADOW",
            "pass": False,
            "detail": "n_panic=0",
        }
    missing = sum(1 for t in panic_tokens if not _has_cast_shadow(t))
    return {
        "id": "SHADOW",
        "pass": missing == 0,
        "detail": f"n_panic={len(panic_tokens)} missing_cast_shadow={missing}",
    }


def check_sse1(
    clean_tokens: list[dict[str, Any]],
    panic_tokens: list[dict[str, Any]],
    *,
    n_cast_sse: int = 0,
) -> CheckResult:
    clean_stats = cast_stats(clean_tokens)
    clean_scored = clean_stats.get("n", 0) > 0
    shadow = check_shadow(panic_tokens)
    return {
        "id": "SSE1",
        "pass": clean_scored and bool(shadow["pass"]),
        "detail": (
            f"n_cast_sse={n_cast_sse} clean_shadow_n={clean_stats.get('n', 0)} "
            f"panic_shadow_ok={shadow['pass']}"
        ),
    }


def score_clean_checks(clean_tokens: list[dict[str, Any]]) -> list[CheckResult]:
    stats = cast_stats(clean_tokens)
    return [check_a1(stats), check_a2(stats), check_a3(stats)]


def score_panic_checks(panic_tokens: list[dict[str, Any]]) -> list[CheckResult]:
    stats = cast_stats(panic_tokens)
    return [
        check_b1(panic_tokens),
        check_b2(stats),
        check_b3(panic_tokens),
        check_b4(panic_tokens),
        check_b5(panic_tokens),
        check_l1(stats),
        check_shadow(panic_tokens),
    ]

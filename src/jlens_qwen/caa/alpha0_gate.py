"""Phase-4 live alpha0 sweep scoring and selection (UNI_CAA01)."""

from __future__ import annotations

import json
import statistics
from typing import Any

from jlens_qwen.caa.cast_math import alpha_eff_mode_a

CLEAN_MEAN_ALPHA_MAX = 0.05
CLEAN_STUB_MEAN_MAX = 0.2
MIN_PANIC_INJECT = 1
MIN_PANIC_G_STREAM = 1
MAX_ALPHA_TOLERANCE = 0.03
MIN_ALPHA_FRACTION = 0.4

CheckResult = dict[str, Any]


def inject_stats(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    casts = [t.get("cast") or {} for t in tokens if t.get("cast")]
    if not casts:
        return {"n": 0}
    alphas = [float(c.get("alpha_eff") or 0) for c in casts]
    applied = [bool(c.get("inject_applied")) for c in casts]
    g_stream = [int(c.get("g_stream") or 0) for c in casts]
    g_recipe = [int(c.get("g_recipe") or 0) for c in casts]
    return {
        "n": len(casts),
        "mean_alpha": statistics.mean(alphas),
        "inject_count": sum(applied),
        "max_alpha": max(alphas) if alphas else 0.0,
        "g_stream_count": sum(g_stream),
        "g_recipe_count": sum(g_recipe),
        "inject_indices": [i for i, c in enumerate(casts) if c.get("inject_applied")],
    }


def _json_demo_value(value: Any) -> Any:
    """Match subprocess demo output (JSON cannot represent tuples)."""
    return json.loads(json.dumps(value))


def demo_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, dict)):
        return _json_demo_value(actual) == _json_demo_value(expected)
    return actual == expected


MAX_ALPHA_CAP_4B = 0.85


def panic_row_passes(
    *,
    alpha_0: float,
    stats: dict[str, Any],
    demo_ok: bool,
    quality_ok: bool = True,
    require_inject: bool = True,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if require_inject:
        if stats.get("inject_count", 0) < MIN_PANIC_INJECT:
            reasons.append(f"inject_count={stats.get('inject_count')}")
        if stats.get("g_stream_count", 0) < MIN_PANIC_G_STREAM:
            reasons.append(f"g_stream_count={stats.get('g_stream_count')}")
        max_alpha = float(stats.get("max_alpha") or 0)
        ceiling = alpha_0 + MAX_ALPHA_TOLERANCE
        if max_alpha > ceiling:
            reasons.append(f"max_alpha={max_alpha:.4f}>{ceiling:.4f}")
        floor = alpha_0 * MIN_ALPHA_FRACTION
        if max_alpha < floor:
            reasons.append(f"max_alpha={max_alpha:.4f}<{floor:.4f}")
    else:
        if stats.get("inject_count", 0) != 0:
            reasons.append(f"inject_count={stats.get('inject_count')}")
    if not demo_ok:
        reasons.append("demo_failed")
    if not quality_ok:
        reasons.append("quality_failed")
    return len(reasons) == 0, reasons


def clean_row_passes(stats: dict[str, Any], *, stub_mean: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if stats.get("inject_count", 0) != 0:
        reasons.append(f"inject_count={stats.get('inject_count')}")
    if float(stats.get("mean_alpha") or 0) > CLEAN_MEAN_ALPHA_MAX:
        reasons.append(f"mean_alpha={stats.get('mean_alpha')}")
    if stub_mean >= CLEAN_STUB_MEAN_MAX:
        reasons.append(f"mean_stub={stub_mean}")
    return len(reasons) == 0, reasons


def select_alpha_0(rows: list[dict[str, Any]]) -> float | None:
    passing = [r for r in rows if r.get("pass")]
    if not passing:
        return None
    return max(float(r["alpha_0"]) for r in passing)


def select_alpha_0_4b(rows: list[dict[str, Any]], *, cap: float = MAX_ALPHA_CAP_4B) -> float | None:
    """Highest passing fit alpha at or below cap (never auto-pick above cap)."""
    passing = sorted({float(r["alpha_0"]) for r in rows if r.get("pass")}, reverse=True)
    if not passing:
        return None
    eligible = [a for a in passing if a <= cap + 1e-9]
    if not eligible:
        return None
    return eligible[0]


def score_sweep_row(
    *,
    alpha_0: float,
    panics: list[dict[str, Any]],
    clean: dict[str, Any],
    require_panic_inject: bool = True,
) -> dict[str, Any]:
    panic_results = []
    all_panic_ok = True
    for p in panics:
        ok, reasons = panic_row_passes(
            alpha_0=alpha_0,
            stats=p["stats"],
            demo_ok=bool(p["demo_ok"]),
            quality_ok=bool(p.get("quality_ok", True)),
            require_inject=require_panic_inject,
        )
        panic_results.append(
            {
                "slug": p["slug"],
                "pass": ok,
                "reasons": reasons,
                "stats": p["stats"],
                "demo_ok": p["demo_ok"],
                "demo_value": p.get("demo_value"),
                "quality_ok": p.get("quality_ok"),
                "quality": p.get("quality"),
                "session": p.get("session"),
                "snapshot": p.get("snapshot"),
            }
        )
        all_panic_ok = all_panic_ok and ok

    clean_ok, clean_reasons = clean_row_passes(
        clean["stats"],
        stub_mean=float(clean.get("stub_mean") or 0),
    )
    row_pass = all_panic_ok and clean_ok
    return {
        "alpha_0": alpha_0,
        "pass": row_pass,
        "panics": panic_results,
        "clean": {
            "pass": clean_ok,
            "reasons": clean_reasons,
            "stats": clean["stats"],
            "stub_mean": clean.get("stub_mean"),
            "session": clean.get("session"),
        },
    }


def _session_invocation(session_json: dict[str, Any] | None) -> dict[str, Any] | None:
    if not session_json:
        return None
    invocations = session_json.get("invocations") or []
    if not invocations:
        return None
    return invocations[0]


def _is_aider_timeout(session_json: dict[str, Any] | None) -> bool:
    if session_json is None:
        return True
    if session_json.get("timed_out"):
        return True
    aider = session_json.get("aider") or {}
    if aider.get("timed_out"):
        return True
    invocations = session_json.get("invocations") or []
    n_inv = int(session_json.get("n_invocations") or len(invocations))
    return n_inv < 1 or not invocations


def _preview_essentially_not_implemented(preview: str) -> bool:
    text = (preview or "").strip()
    if not text:
        return True
    if "main.py" in text or "```" in text:
        return False
    normalized = text.replace(":", "").strip()
    if normalized == "NotImplementedError":
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return len(lines) <= 2 and all("NotImplementedError" in ln for ln in lines)


def _preview_has_codegen(preview: str) -> bool:
    text = preview or ""
    return "main.py" in text or "```" in text


def _garbled_reasons_contain_not_implemented(quality: dict[str, Any]) -> bool:
    reasons = quality.get("reasons") or []
    return any("NotImplementedError" in str(r) for r in reasons)


def _is_other_garbled(quality: dict[str, Any]) -> bool:
    if quality.get("garbled_ok", True):
        return False
    if _garbled_reasons_contain_not_implemented(quality):
        return False
    return True


def classify_failure(row: dict[str, Any], session_json: dict[str, Any] | None) -> str:
    """First-match-wins failure taxonomy for Phase 4b panic rows."""
    if _is_aider_timeout(session_json):
        return "aider_timeout"

    inv = _session_invocation(session_json)
    completion_tokens = int(inv.get("completion_tokens") or 0) if inv else 0
    assistant_preview = str(inv.get("assistant_preview") or "") if inv else ""

    if completion_tokens <= 20 and _preview_essentially_not_implemented(assistant_preview):
        return "no_output"

    quality = row.get("quality") or {}
    if _garbled_reasons_contain_not_implemented(quality) and _preview_has_codegen(
        assistant_preview
    ):
        return "stub_left"

    if _is_other_garbled(quality):
        return "garbled"

    if quality.get("garbled_ok", True) and not quality.get("pytest_ok", True):
        return "pytest_fail"

    if row.get("quality_ok", True) and not row.get("demo_ok", True):
        return "demo_mismatch"

    require_inject = bool(row.get("require_inject", True))
    stats = row.get("stats") or {}
    if require_inject and int(stats.get("inject_count") or 0) < MIN_PANIC_INJECT:
        return "cast_no_inject"

    alpha_0 = float(row.get("alpha_0") or 0)
    max_alpha = float(stats.get("max_alpha") or 0)
    if max_alpha > alpha_0 + MAX_ALPHA_TOLERANCE:
        return "cast_over_inject"

    if row.get("pass"):
        return "pass"

    return "unknown"


def sabotage_alpha_scales() -> dict[str, Any]:
    base = alpha_eff_mode_a(
        alpha_0=0.5,
        w_sev=1.0,
        gamma_val=1.0,
        g_recipe=1,
        g_stream=1,
    )
    high = alpha_eff_mode_a(
        alpha_0=0.85,
        w_sev=1.0,
        gamma_val=1.0,
        g_recipe=1,
        g_stream=1,
    )
    ratio = high / base if base > 0 else 0.0
    expected = 0.85 / 0.5
    return {
        "id": "A0-S1",
        "pass": abs(ratio - expected) < 0.01,
        "detail": f"ratio={ratio:.4f} expected={expected:.4f}",
    }

#!/usr/bin/env python
"""Validate Method B calm/bad pair — text gates + chaos gates (UNI_CAA01 practice)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from caa_error_combo_study import score_text  # noqa: E402
from jlens_qwen.caa.method_b_databus import validate_method_b_record  # noqa: E402
from prefix_divergence_tracer import (  # noqa: E402
    PANIC_RECIPE,
    eligible_for_panic_trace,
    trace_invocation,
)

TailKind = Literal["calm", "bad"]

ELIGIBLE_FLAG_RATE = 0.01
CHAOS_DEBOUNCE_STEPS = 15


def _tail_flag_rates(chaos: dict[str, Any], from_step: int) -> tuple[float, float]:
    tokens = chaos.get("tokens") or []
    tail = [t for t in tokens if int(t.get("step", -1)) >= from_step]
    if not tail:
        return 0.0, 0.0
    n = len(tail)
    d_n = sum(1 for t in tail if "B_distress" in (t.get("flags") or []))
    s_n = sum(1 for t in tail if "B_stub" in (t.get("flags") or []))
    return d_n / n, s_n / n


def _chaos_panic_near_d(traced: dict[str, Any], expected_d: int | None) -> bool:
    """Teacher-forced replay: panic masses often lag text divergence by a few steps."""
    if expected_d is None:
        return False
    d_lex = traced.get("d_lexicon")
    d_flg = traced.get("d_flags")
    d_dist = traced.get("d_distress_only")
    d_stub = traced.get("d_stub_only")
    candidates = [x for x in (d_lex, d_flg, d_dist, d_stub) if x is not None]
    if not candidates:
        return False
    chaos_d = min(candidates) if len(candidates) >= 2 else candidates[0]
    return int(expected_d) <= int(chaos_d) <= int(expected_d) + CHAOS_DEBOUNCE_STEPS


def _recipe_fired(chaos: dict[str, Any], recipe_id: str = PANIC_RECIPE) -> bool:
    timeline = (chaos.get("monitor") or {}).get("recipe_timeline") or []
    return any(ev.get("recipe_id") == recipe_id for ev in timeline)


def _recipe_combo_fired(chaos: dict[str, Any]) -> bool:
    """B_STUB_DISTRESS or (B_DISTRESS AND B_STUB) both in timeline."""
    if _recipe_fired(chaos):
        return True
    timeline = (chaos.get("monitor") or {}).get("recipe_timeline") or []
    has_d = any(ev.get("recipe_id") == "B_DISTRESS" for ev in timeline)
    has_s = any(ev.get("recipe_id") == "B_STUB" for ev in timeline)
    return has_d and has_s


def validate_tail(
    *,
    tail_kind: TailKind,
    assistant_text: str,
    chaos: dict[str, Any],
    inv: dict[str, Any],
    expected_d: int | None,
    experiment: str = "",
    task_slug: str = "",
    prefix_token_ids: list[int] | None = None,
    tail_text: str | None = None,
) -> dict[str, Any]:
    """Gate one forced replay (calm or bad).

    Calm admission adds T8–T11 (stricter than T1–T6 plumbing gates).
    """
    checks: list[dict[str, Any]] = []
    scores = score_text(assistant_text)
    traced = trace_invocation(
        inv,
        experiment=experiment,
        task_slug=task_slug,
        use_tokenizer=False,
    ).to_dict()
    eligible = eligible_for_panic_trace(chaos)
    hints = chaos.get("divergence_hints") or {}

    if tail_kind == "calm":
        checks.append(
            {
                "id": "T1_calm_stub_score",
                "pass": scores["stub_score"] == 0,
                "detail": f"stub_score={scores['stub_score']} hits={scores['stub_hits']}",
            }
        )
        checks.append(
            {
                "id": "T2_calm_distress_score",
                "pass": scores["distress_score"] == 0,
                "detail": f"distress_score={scores['distress_score']} hits={scores['distress_hits']}",
            }
        )
        checks.append(
            {
                "id": "T3_calm_not_eligible",
                "pass": not eligible,
                "detail": f"eligible={eligible}",
            }
        )
        checks.append(
            {
                "id": "T4_calm_no_panic_recipe",
                "pass": not _recipe_fired(chaos),
                "detail": f"recipe_fired={_recipe_fired(chaos)}",
            }
        )
        checks.append(
            {
                "id": "T5_calm_no_d_consensus",
                "pass": traced.get("d_consensus") is None,
                "detail": f"d_consensus={traced.get('d_consensus')}",
            }
        )
        b_d = float(hints.get("b_distress_rate", 0))
        b_s = float(hints.get("b_stub_rate", 0))
        checks.append(
            {
                "id": "T6_calm_low_flag_rates",
                "pass": b_d < ELIGIBLE_FLAG_RATE and b_s < ELIGIBLE_FLAG_RATE,
                "detail": f"b_distress_rate={b_d:.4f} b_stub_rate={b_s:.4f}",
            }
        )
        # T8: score calm *tail* text after d (or full if no split provided)
        t8_text = tail_text if tail_text is not None else assistant_text
        t8_scores = score_text(t8_text)
        checks.append(
            {
                "id": "T8_calm_tail_text_clean",
                "pass": t8_scores["stub_score"] == 0 and t8_scores["distress_score"] == 0,
                "detail": (
                    f"stub={t8_scores['stub_score']} distress={t8_scores['distress_score']} "
                    f"hits_stub={t8_scores['stub_hits']} hits_distress={t8_scores['distress_hits']}"
                ),
            }
        )
        # T9: zero distress/stub rates for steps >= d
        tail_d_rate, tail_s_rate = _tail_flag_rates(chaos, expected_d or 0)
        checks.append(
            {
                "id": "T9_calm_tail_rates_zero",
                "pass": tail_d_rate == 0.0 and tail_s_rate == 0.0,
                "detail": f"tail_b_distress={tail_d_rate:.4f} tail_b_stub={tail_s_rate:.4f}",
            }
        )
        # T10: forced replay token_ids[:d] == prefix_token_ids
        if prefix_token_ids is not None and expected_d is not None:
            replay_ids = list(chaos.get("token_ids") or [])[: int(expected_d)]
            checks.append(
                {
                    "id": "T10_calm_prefix_token_eq",
                    "pass": replay_ids == list(prefix_token_ids)[: int(expected_d)],
                    "detail": (
                        f"replay_len={len(replay_ids)} prefix_len={len(prefix_token_ids)} "
                        f"d={expected_d}"
                    ),
                }
            )
        else:
            checks.append(
                {
                    "id": "T10_calm_prefix_token_eq",
                    "pass": False,
                    "detail": "prefix_token_ids required for calm admission — fail closed",
                }
            )
        # T11: no B_STUB_DISTRESS and no B_DISTRESS+B_STUB combo
        checks.append(
            {
                "id": "T11_calm_no_panic_combo",
                "pass": not _recipe_combo_fired(chaos),
                "detail": f"combo={_recipe_combo_fired(chaos)}",
            }
        )
    else:
        tail_text_for_score = tail_text if tail_text is not None else assistant_text
        tail_scores = score_text(tail_text_for_score)
        scores_for_gates = tail_scores if tail_text is not None else scores
        tail_d_rate, tail_s_rate = _tail_flag_rates(chaos, expected_d or 0)
        hints = chaos.get("divergence_hints") or {}
        hint_d = float(hints.get("b_distress_rate", 0))
        hint_s = float(hints.get("b_stub_rate", 0))
        has_stub_chaos = (
            tail_s_rate >= ELIGIBLE_FLAG_RATE
            or hint_s >= ELIGIBLE_FLAG_RATE
            or any(
                ev.get("recipe_id") in {"B_STUB", "B_STUB_DISTRESS", PANIC_RECIPE}
                for ev in (chaos.get("monitor") or {}).get("recipe_timeline") or []
            )
        )
        checks.append(
            {
                "id": "T1_bad_stub_score",
                "pass": scores_for_gates["stub_score"] >= 1 or has_stub_chaos,
                "detail": (
                    f"stub_score={scores_for_gates['stub_score']} hits={scores_for_gates['stub_hits']} "
                    f"chaos_stub={has_stub_chaos}"
                ),
            }
        )
        has_distress_chaos = (
            eligible
            or tail_d_rate >= ELIGIBLE_FLAG_RATE
            or hint_d >= ELIGIBLE_FLAG_RATE
            or any(
                ev.get("recipe_id") == "B_DISTRESS"
                for ev in (chaos.get("monitor") or {}).get("recipe_timeline") or []
            )
        )
        checks.append(
            {
                "id": "T2_bad_distress_score",
                "pass": scores_for_gates["distress_score"] >= 1 or has_distress_chaos,
                "detail": (
                    f"distress_score={scores_for_gates['distress_score']} "
                    f"hits={scores_for_gates['distress_hits']} chaos_distress={has_distress_chaos}"
                ),
            }
        )
        has_both_estimators = (
            traced.get("d_distress_only") is not None and traced.get("d_stub_only") is not None
        )
        chaos_ok = eligible or (
            tail_d_rate >= ELIGIBLE_FLAG_RATE
            and tail_s_rate >= ELIGIBLE_FLAG_RATE * 0.5
        ) or (
            hint_d >= ELIGIBLE_FLAG_RATE and hint_s >= ELIGIBLE_FLAG_RATE * 0.5
        ) or has_both_estimators
        checks.append(
            {
                "id": "T3_bad_eligible_or_tail_flags",
                "pass": chaos_ok,
                "detail": (
                    f"eligible={eligible} tail_b_distress={tail_d_rate:.4f} "
                    f"tail_b_stub={tail_s_rate:.4f} hint_d={hint_d:.4f} hint_s={hint_s:.4f}"
                ),
            }
        )
        recipe_combo = _recipe_fired(chaos) or (
            any(
                ev.get("recipe_id") == "B_DISTRESS"
                for ev in (chaos.get("monitor") or {}).get("recipe_timeline") or []
            )
            and any(
                ev.get("recipe_id") == "B_STUB"
                for ev in (chaos.get("monitor") or {}).get("recipe_timeline") or []
            )
        )
        checks.append(
            {
                "id": "T4_bad_panic_recipe",
                "pass": recipe_combo,
                "detail": f"B_STUB_DISTRESS={_recipe_fired(chaos)} combo={recipe_combo}",
            }
        )
        if expected_d is not None:
            d = traced.get("d_consensus")
            near = _chaos_panic_near_d(traced, expected_d)
            checks.append(
                {
                    "id": "T5_bad_d_near_text_d",
                    "pass": (d is not None and abs(int(d) - int(expected_d)) <= CHAOS_DEBOUNCE_STEPS)
                    or near,
                    "detail": (
                        f"d_consensus={d} expected_d={expected_d} "
                        f"d_lex={traced.get('d_lexicon')} d_flags={traced.get('d_flags')}"
                    ),
                }
            )
            token_ids = (chaos.get("token_ids") or [])[: int(expected_d)]
            prefix_len = len(traced.get("prefix_token_ids") or token_ids)
            checks.append(
                {
                    "id": "T6_bad_prefix_len",
                    "pass": prefix_len == int(expected_d),
                    "detail": f"prefix_len={prefix_len} expected_d={expected_d}",
                }
            )
        checks.append(
            {
                "id": "T7_bad_confidence",
                "pass": traced.get("confidence") in ("low", "medium", "high")
                or chaos_ok,
                "detail": f"confidence={traced.get('confidence')}",
            }
        )

    ok = all(c["pass"] for c in checks)
    return {
        "tail_kind": tail_kind,
        "pass": ok,
        "checks": checks,
        "scores": scores,
        "traced": traced,
        "eligible": eligible,
    }


def validate_pair(
    *,
    spec: dict[str, Any],
    calm_run: dict[str, Any],
    bad_run: dict[str, Any],
    prompt_len: int,
) -> dict[str, Any]:
    """Validate full Method B pair including prefix match and databus records."""
    checks: list[dict[str, Any]] = []

    calm_prefix = calm_run.get("prefix_token_ids") or []
    bad_prefix = bad_run.get("prefix_token_ids") or []
    spec_prefix = list(spec.get("prefix_token_ids") or [])
    expected_d = int(spec.get("d", 0))

    calm_ids = (calm_run.get("chaos") or {}).get("token_ids") or []
    bad_ids = (bad_run.get("chaos") or {}).get("token_ids") or []
    if not calm_prefix and calm_ids and expected_d > 0:
        calm_prefix = list(calm_ids[:expected_d])
    if not bad_prefix and bad_ids and expected_d > 0:
        bad_prefix = list(bad_ids[:expected_d])

    checks.append(
        {
            "id": "P1_prefix_match",
            "pass": calm_prefix == bad_prefix == spec_prefix and len(spec_prefix) == expected_d,
            "detail": (
                f"calm={len(calm_prefix)} bad={len(bad_prefix)} "
                f"spec={len(spec_prefix)} d={expected_d}"
            ),
        }
    )
    checks.append(
        {
            "id": "P2_tails_differ_at_d",
            "pass": (spec.get("calm_tail_token_ids") or [])[:1] != (spec.get("bad_tail_token_ids") or [])[:1],
            "detail": "first tail token must differ",
        }
    )
    checks.append(
        {
            "id": "P3_d_positive",
            "pass": expected_d > 0,
            "detail": f"d={expected_d}",
        }
    )

    calm_mb = calm_run.get("method_b_record")
    bad_mb = bad_run.get("method_b_record")
    for kind, mb in (("calm", calm_mb), ("bad", bad_mb)):
        if not mb:
            checks.append(
                {
                    "id": f"P4_{kind}_databus_present",
                    "pass": False,
                    "detail": "method_b_record missing",
                }
            )
            continue
        v = validate_method_b_record(mb, prompt_len=prompt_len)
        checks.append(
            {
                "id": f"P4_{kind}_databus",
                "pass": v["pass"],
                "detail": v["checks"],
            }
        )
    if expected_d > 0:
        for kind, mb in (("calm", calm_mb), ("bad", bad_mb)):
            if mb:
                checks.append(
                    {
                        "id": f"P6_{kind}_d_matches_spec",
                        "pass": int(mb.get("d", -1)) == expected_d,
                        "detail": f"record d={mb.get('d')} spec d={expected_d}",
                    }
                )

    calm_tail = validate_tail(
        tail_kind="calm",
        assistant_text=str(calm_run.get("assistant_text") or ""),
        chaos=calm_run.get("chaos") or {},
        inv=calm_run.get("invocation") or {},
        expected_d=expected_d,
        experiment=str(calm_run.get("experiment") or ""),
        task_slug=str(calm_run.get("task_slug") or ""),
        prefix_token_ids=list(spec_prefix) if spec_prefix else None,
        tail_text=str(
            calm_run.get("tail_text")
            or spec.get("calm_tail_text")
            or ""
        )
        or None,
    )
    bad_tail = validate_tail(
        tail_kind="bad",
        assistant_text=str(bad_run.get("assistant_text") or ""),
        chaos=bad_run.get("chaos") or {},
        inv=bad_run.get("invocation") or {},
        expected_d=expected_d,
        experiment=str(bad_run.get("experiment") or ""),
        task_slug=str(bad_run.get("task_slug") or ""),
        tail_text=str(spec.get("bad_tail_text") or "") or None,
    )

    ok = all(c["pass"] for c in checks) and calm_tail["pass"] and bad_tail["pass"]
    return {
        "pass": ok,
        "pair_checks": checks,
        "calm": calm_tail,
        "bad": bad_tail,
    }

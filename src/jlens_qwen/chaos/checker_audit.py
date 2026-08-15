"""Per-invocation A–G checker audit for dashboard and strict gates."""

from __future__ import annotations

from typing import Any

from jlens_qwen.chaos.constants import F2_STRIDE
from jlens_qwen.chaos.flags import THRESH, evaluate_flags


def _flag_rate(tokens: list[dict], prefix: str) -> float:
    if not tokens:
        return 0.0
    n = sum(1 for t in tokens if any(f.startswith(prefix) for f in (t.get("flags") or [])))
    return n / len(tokens)


def _max_metric(tokens: list[dict], path: tuple[str, str]) -> float:
    key, sub = path
    vals = [float((t.get(key) or {}).get(sub, 0)) for t in tokens]
    return max(vals) if vals else 0.0


def audit_invocation(chaos: dict[str, Any] | None) -> dict[str, Any]:
    """Return A–G checker status for one invocation's chaos record."""
    if not chaos:
        return {"available": False, "checkers": {}}

    tokens = chaos.get("tokens") or []
    events = chaos.get("events") or []
    curves = chaos.get("curves_E") or {}
    timing = chaos.get("timing") or {}
    composites = chaos.get("composites") or {}

    # --- A: distribution shape ---
    a_high_ent = _flag_rate(tokens, "A_high_entropy")
    a_low_mar = _flag_rate(tokens, "A_low_margin")
    a_fired = a_high_ent > 0.05 or a_low_mar > 0.05
    a_ok = bool(tokens) and _max_metric(tokens, ("A_vocab", "entropy")) > 0

    # --- B: token-class hygiene ---
    b_vocab_f = _flag_rate(tokens, "B_")
    b_transport_layers = 0
    b_jlens_layers = 0
    for t in tokens:
        for pdata in (t.get("probes") or {}).values():
            if pdata.get("B_transport"):
                b_transport_layers += 1
            if pdata.get("B_jlens"):
                b_jlens_layers += 1
    b_ok = bool(tokens) and "code_mass" in (tokens[0].get("B_vocab") or {})

    # --- C: trajectory ---
    c_thrash = _flag_rate(tokens, "C_thrash")
    c_osc = _flag_rate(tokens, "C_oscillation")
    # Legacy runs stored old flag logic; re-evaluate with calibrated rules when possible
    sim_flags: set[str] = set()
    for t in tokens:
        payload = {k: v for k, v in t.items() if k not in ("flags", "step", "token_id")}
        if t.get("D") and not payload.get("D_boundary"):
            payload.pop("D", None)
        payload.pop("D_roll", None)
        if t.get("step") == 0 and payload.get("D"):
            payload["D_boundary"] = True
        sim_flags.update(evaluate_flags(payload))
    c_thrash_sim = sum(1 for f in sim_flags if f.startswith("C_")) > 0
    settling = -1.0
    for t in reversed(tokens):
        tau = float((t.get("C_vocab") or {}).get("settling_tau", -1))
        if tau >= 0:
            settling = tau
            break
    c_ok = bool(tokens)

    # --- D: cross-position (boundary events only) ---
    gen_start = next((e for e in events if e.get("kind") == "gen_start"), {})
    d_field = gen_start.get("D") or {}
    d_boundary_flags: list[str] = []
    for t in tokens:
        if t.get("D_boundary"):
            d_boundary_flags.extend(t.get("flags") or [])
    d_fired = any(f.startswith("D_") for f in d_boundary_flags) or abs(
        float(d_field.get("delta_H", 0))
    ) >= THRESH["D_focus_break_abs_dH"] or float(d_field.get("js_div", 0)) >= THRESH["D_focus_break_js"]
    d_ok = "prompt_end" in [e.get("kind") for e in events] and bool(gen_start)

    # --- E: layer curves ---
    cliff = curves.get("cliff_layer")
    e_ok = cliff is not None and 8 <= int(cliff) <= 26
    series = curves.get("series") or {}

    # --- F: J-lens vs sampling-head disagreement (not naive logit-lens) ---
    f_events = [e for e in events if e.get("kind") == "F_disagreement"]
    min_overlap_sample = 1.0
    min_overlap_naive = 1.0
    for t in tokens:
        for pdata in (t.get("probes") or {}).values():
            if not isinstance(pdata, dict):
                continue
            f = pdata.get("F")
            if not isinstance(f, dict):
                continue
            if f.get("overlap_sample") is not None:
                min_overlap_sample = min(
                    min_overlap_sample, float(f["overlap_sample"])
                )
            if f.get("overlap_naive") is not None:
                min_overlap_naive = min(min_overlap_naive, float(f["overlap_naive"]))
    for ev in f_events:
        for layer_f in (ev.get("layers") or {}).values():
            if isinstance(layer_f, dict):
                if layer_f.get("overlap_sample") is not None:
                    min_overlap_sample = min(
                        min_overlap_sample, float(layer_f["overlap_sample"])
                    )
                if layer_f.get("overlap_naive") is not None:
                    min_overlap_naive = min(
                        min_overlap_naive, float(layer_f["overlap_naive"])
                    )
    has_sample_metric = min_overlap_sample < 1.0
    f_ok = bool(f_events) or has_sample_metric or any(
        isinstance(pdata, dict) and isinstance(pdata.get("F"), dict)
        for t in tokens
        for pdata in (t.get("probes") or {}).values()
    )
    f_alarm = has_sample_metric and min_overlap_sample <= THRESH["F_low_overlap"]

    # --- G: sparse residual ---
    g_events = [e for e in events if e.get("kind") == "G_sparse"]
    max_relmse = 0.0
    for ev in g_events:
        for layer_g in (ev.get("layers") or {}).values():
            max_relmse = max(max_relmse, float(layer_g.get("relMSE", 0)))
    g_ok = bool(g_events)
    g_alarm = g_ok and max_relmse >= THRESH["G_high_relmse"]

    checkers = {
        "A": {
            "label": "Distribution shape",
            "active": a_ok,
            "fired": a_fired,
            "severity": "warn" if a_fired else "ok",
            "detail": f"low_margin={a_low_mar:.0%} high_entropy={a_high_ent:.0%}",
            "metrics": {
                "max_entropy": _max_metric(tokens, ("A_vocab", "entropy")),
                "min_margin": min(
                    (float((t.get("A_vocab") or {}).get("margin", 1)) for t in tokens),
                    default=1.0,
                ),
            },
        },
        "B_vocab": {
            "label": "Vocab class hygiene",
            "active": b_ok,
            "fired": _flag_rate(tokens, "B_distress") > 0
            or _flag_rate(tokens, "B_stub") > 0
            or _max_metric(tokens, ("B_vocab", "distress_mass")) >= THRESH["B_distress"],
            "severity": (
                "warn"
                if _max_metric(tokens, ("B_vocab", "distress_mass")) >= THRESH["B_distress"]
                else "ok"
            ),
            "detail": (
                f"distress={_max_metric(tokens, ('B_vocab', 'distress_mass')):.3f} "
                f"stub={_max_metric(tokens, ('B_vocab', 'stub_mass')):.3f}"
            ),
        },
        "B_transport": {
            "label": "Transport prototypes",
            "active": b_transport_layers > 0,
            "fired": False,
            "severity": "ok" if b_transport_layers > 0 else "fail",
            "detail": f"rows_with_L16_transport={b_transport_layers}/{len(tokens)}",
        },
        "B_jlens": {
            "label": "J-lens top-k classes",
            "active": b_jlens_layers > 0 or any(e.get("kind") == "prompt_end" for e in events),
            "fired": False,
            "severity": "ok" if b_jlens_layers > 0 else "info",
            "detail": f"F2/F3 jlens rows={b_jlens_layers}",
        },
        "C": {
            "label": "Trajectory stability",
            "active": c_ok,
            "fired": c_thrash_sim or c_osc > 0.05,
            "severity": "warn" if c_thrash_sim else ("ok" if c_thrash < 0.1 else "info"),
            "detail": (
                f"stored_thrash={c_thrash:.0%} osc={c_osc:.0%} "
                f"calibrated_C_alarm={c_thrash_sim}"
            ),
        },
        "D": {
            "label": "Cross-position contrast",
            "active": d_ok,
            "fired": d_fired,
            "severity": "warn" if d_fired else "ok",
            "detail": (
                f"ΔH={float(d_field.get('delta_H', 0)):.3f} "
                f"JS={float(d_field.get('js_div', 0)):.3f} "
                f"|ΔM_code|={abs(float(d_field.get('delta_M_code', 0))):.3f}"
            ),
        },
        "E": {
            "label": "Layer curve (cliff)",
            "active": e_ok,
            "fired": False,
            "severity": "ok" if e_ok else "fail",
            "detail": f"cliff_layer=L{cliff}" if cliff is not None else "no cliff",
            "metrics": {"cliff_mag": curves.get("cliff_mag"), "series_keys": list(series.keys())},
        },
        "F": {
            "label": "J-lens vs sampling-head disagreement",
            "active": f_ok,
            "fired": f_alarm,
            "severity": "warn" if f_alarm else ("ok" if f_ok else "info"),
            "detail": (
                f"events={len(f_events)} "
                f"min_overlap_sample={min_overlap_sample:.2f} "
                f"min_overlap_naive={min_overlap_naive:.2f}"
            ),
        },
        "G": {
            "label": "Sparse residual",
            "active": g_ok,
            "fired": g_alarm,
            "severity": "warn" if g_alarm else ("ok" if g_ok else "info"),
            "detail": f"events={len(g_events)} max_relMSE={max_relmse:.3f}",
        },
    }

    return {
        "available": True,
        "version": chaos.get("version", 1),
        "n_tokens": len(tokens),
        "composites": composites,
        "timing_p95_ms": float(timing.get("chaos_ms_per_token_p95", 0)),
        "degraded_mode": bool(chaos.get("degraded_mode")),
        "f2_stride": chaos.get("f2_stride", F2_STRIDE),
        "checkers": checkers,
        "all_active": all(c["active"] for c in checkers.values() if c["label"]),
        "panic_like": sum(1 for c in checkers.values() if c.get("severity") == "warn"),
    }

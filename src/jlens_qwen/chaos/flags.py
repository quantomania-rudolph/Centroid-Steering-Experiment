"""Flag thresholds and composite indices PLDS / FCI / SRS."""

from __future__ import annotations

from typing import Any

# Fixed v1 thresholds (tunable later via YAML)
THRESH = {
    "A_high_entropy": 2.5,  # nats on top-20
    "A_low_margin": 0.15,
    "B_distress": 0.12,
    "B_stub": 0.10,
    "B_junk": 0.15,
    "B_low_code": 0.05,
    "C_thrash_flip": 0.55,
    "C_oscillation": 0.35,
    "D_focus_break_abs_dH": 0.8,
    "D_focus_break_js": 0.4,
    "F_low_overlap": 0.4,
    "G_high_relmse": 0.05,
    "E_cliff_mag": 0.35,
}


def e_cliff_alarm(cliff_layer: int | None, cliff_mag: float) -> bool:
    """True when entropy cliff is in probe range and magnitude exceeds threshold."""
    if cliff_layer is None:
        return False
    if not (8 <= int(cliff_layer) <= 26):
        return False
    return abs(float(cliff_mag)) >= THRESH["E_cliff_mag"]


def evaluate_flags(token_payload: dict[str, Any]) -> list[str]:
    """Return flag category names firing on this token payload.

    Calibrated to avoid always-on false positives during normal codegen:
    - C_thrash requires oscillation or never-settling high flip
    - B_low_code fires when code_mass in top-k is very low
    - A_low_margin requires both low margin AND elevated entropy
    - D only evaluated on boundary payloads (D_boundary=True)
    """
    flags: list[str] = []
    a = token_payload.get("A_vocab") or {}
    ent = float(a.get("entropy", 0))
    margin = float(a.get("margin", 1))
    if ent >= THRESH["A_high_entropy"]:
        flags.append("A_high_entropy")
    if margin <= THRESH["A_low_margin"] and ent >= 1.2:
        flags.append("A_low_margin")

    b = token_payload.get("B_vocab") or {}
    if float(b.get("distress_mass", 0)) >= THRESH["B_distress"]:
        flags.append("B_distress")
    if float(b.get("stub_mass", 0)) >= THRESH["B_stub"]:
        flags.append("B_stub")
    if float(b.get("junk_mass", 0)) >= THRESH["B_junk"]:
        flags.append("B_junk")
    if float(b.get("code_mass", 0)) <= THRESH["B_low_code"]:
        flags.append("B_low_code")

    c = token_payload.get("C_vocab") or {}
    flip = float(c.get("flip_rate", 0))
    osc = float(c.get("oscillation", 0))
    # C_thrash: alternation pattern only (not mere token change each step)
    if osc >= THRESH["C_oscillation"] and flip >= 0.4:
        flags.append("C_oscillation")
    if osc >= max(THRESH["C_oscillation"], 0.4) and flip >= THRESH["C_thrash_flip"]:
        flags.append("C_thrash")

    if token_payload.get("D_boundary"):
        d = token_payload.get("D") or {}
        if abs(float(d.get("delta_H", 0))) >= THRESH["D_focus_break_abs_dH"]:
            flags.append("D_focus_break")
        if float(d.get("js_div", 0)) >= THRESH["D_focus_break_js"]:
            flags.append("D_focus_break")

    # dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def flag_categories(flags: list[str]) -> set[str]:
    """Map fine flags to coarse categories A/B/C/D for G gate."""
    cats: set[str] = set()
    for f in flags:
        if f.startswith("A_"):
            cats.add("A")
        elif f.startswith("B_"):
            cats.add("B")
        elif f.startswith("C_"):
            cats.add("C")
        elif f.startswith("D_"):
            cats.add("D")
    return cats


def composite_plds(token_rows: list[dict[str, Any]]) -> float:
    """Per-Layer Distress Score proxy: mean of entropy + (1-code) + flip + distress."""
    if not token_rows:
        return 0.0
    scores = []
    for row in token_rows:
        a = row.get("A_vocab") or {}
        b = row.get("B_vocab") or {}
        c = row.get("C_vocab") or {}
        h = float(a.get("entropy", 0))
        code = float(b.get("code_mass", 0))
        dist = float(b.get("distress_mass", 0))
        flip = float(c.get("flip_rate", 0))
        scores.append(0.25 * h + 0.25 * (1.0 - code) + 0.25 * flip + 0.25 * dist)
    return sum(scores) / len(scores)


def composite_fci(token_rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> float:
    """Focus Clarity Index — higher = worse focus (more chaos)."""
    if not token_rows:
        return 0.0
    ent = [float((r.get("A_vocab") or {}).get("entropy", 0)) for r in token_rows]
    mean_h = sum(ent) / len(ent)
    settling = -1.0
    for r in reversed(token_rows):
        tau = float((r.get("C_vocab") or {}).get("settling_tau", -1))
        if tau >= 0:
            settling = tau
            break
    settle_pen = settling / max(len(token_rows), 1) if settling >= 0 else 1.0
    return mean_h + settle_pen


def composite_srs(token_rows: list[dict[str, Any]]) -> float:
    """Stub Risk Score."""
    if not token_rows:
        return 0.0
    stub = [float((r.get("B_vocab") or {}).get("stub_mass", 0)) for r in token_rows]
    dist = [float((r.get("B_vocab") or {}).get("distress_mass", 0)) for r in token_rows]
    code = [float((r.get("B_vocab") or {}).get("code_mass", 0)) for r in token_rows]
    return (
        0.4 * (sum(stub) / len(stub))
        + 0.4 * (sum(dist) / len(dist))
        + 0.2 * (1.0 - sum(code) / len(code))
    )

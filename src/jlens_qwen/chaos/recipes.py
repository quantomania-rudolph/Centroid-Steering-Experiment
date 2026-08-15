"""RecipeMatcher — bitmask combination of A–G atoms into steering recipe IDs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from jlens_qwen.chaos.metrics_f import f_alarm_from_metrics

Severity = Literal["none", "tip", "warn", "critical"]

ATOM_KEYS: tuple[str, ...] = (
    "A_uncertain",
    "B_distress",
    "B_stub",
    "B_junk",
    "C_osc",
    "D_break",
    "E_cliff",
    "F_disagree",
    "G_oob",
)

DEBOUNCE_TOKENS = 8

# Bit positions for fast lookup (golden tests use these)
ATOM_BITS: dict[str, int] = {name: 1 << i for i, name in enumerate(ATOM_KEYS)}


@dataclass(frozen=True)
class RecipeDef:
    recipe_id: str
    required: frozenset[str]
    any_b: bool = False
    severity: Severity = "tip"
    alpha_hint: float = 0.5
    layer_default: int = 16
    priority: int = 0
    caa_tags: tuple[str, ...] = ()


def _b_atoms() -> frozenset[str]:
    return frozenset({"B_distress", "B_stub", "B_junk"})


# Priority: higher number wins when multiple recipes match
RECIPE_TABLE: tuple[RecipeDef, ...] = (
    RecipeDef(
        "PANIC_FULL",
        frozenset({"A_uncertain", "C_osc"}),
        any_b=True,
        severity="critical",
        alpha_hint=1.0,
        priority=100,
        caa_tags=("A_UNCERTAIN", "B_STUB_DISTRESS", "C_OSC"),
    ),
    RecipeDef(
        "STRUCT_BREAK",
        frozenset({"D_break", "E_cliff", "F_disagree"}),
        severity="critical",
        alpha_hint=1.0,
        priority=95,
        caa_tags=("D_FOCUS", "E_CLIFF", "F_LENS"),
    ),
    RecipeDef(
        "AGENT_FLAIL",
        frozenset({"B_stub", "B_distress", "F_disagree"}),
        severity="critical",
        alpha_hint=1.0,
        priority=90,
        caa_tags=("B_STUB_DISTRESS", "F_LENS"),
    ),
    RecipeDef(
        "B_STUB_DISTRESS",
        frozenset({"B_stub", "B_distress"}),
        severity="warn",
        alpha_hint=0.75,
        priority=80,
        caa_tags=("B_STUB_DISTRESS",),
    ),
    RecipeDef(
        "A_C_VACILLATE",
        frozenset({"A_uncertain", "C_osc"}),
        severity="warn",
        alpha_hint=0.5,
        priority=70,
        caa_tags=("A_UNCERTAIN", "C_OSC"),
    ),
    RecipeDef(
        "A_D_FOCUS",
        frozenset({"A_uncertain", "D_break"}),
        severity="warn",
        alpha_hint=0.5,
        priority=65,
        caa_tags=("A_UNCERTAIN", "D_FOCUS"),
    ),
    RecipeDef(
        "B_F_SURFACE",
        frozenset({"F_disagree"}),
        any_b=True,
        severity="warn",
        alpha_hint=0.75,
        priority=60,
        caa_tags=("B_DISTRESS", "F_LENS"),
    ),
    RecipeDef(
        "F_E_DEPTH",
        frozenset({"F_disagree", "E_cliff"}),
        severity="warn",
        alpha_hint=0.75,
        priority=55,
        caa_tags=("F_LENS", "E_CLIFF"),
    ),
    RecipeDef(
        "D_F_READOUT",
        frozenset({"D_break", "F_disagree"}),
        severity="warn",
        alpha_hint=0.75,
        priority=50,
        caa_tags=("D_FOCUS", "F_LENS"),
    ),
    RecipeDef(
        "C_F_TRAJ",
        frozenset({"C_osc", "F_disagree"}),
        severity="warn",
        alpha_hint=0.75,
        priority=45,
        caa_tags=("C_OSC", "F_LENS"),
    ),
    RecipeDef(
        "B_G_RESID",
        frozenset({"G_oob"}),
        any_b=True,
        severity="warn",
        alpha_hint=0.5,
        priority=40,
        caa_tags=("G_OOB", "B_DISTRESS"),
    ),
    RecipeDef(
        "A_UNCERTAIN",
        frozenset({"A_uncertain"}),
        severity="tip",
        alpha_hint=0.5,
        priority=10,
        caa_tags=("A_UNCERTAIN",),
    ),
    RecipeDef(
        "B_DISTRESS",
        frozenset({"B_distress"}),
        severity="warn",
        alpha_hint=0.75,
        priority=15,
        caa_tags=("B_DISTRESS",),
    ),
    RecipeDef(
        "B_STUB",
        frozenset({"B_stub"}),
        severity="warn",
        alpha_hint=0.75,
        priority=15,
        caa_tags=("B_STUB",),
    ),
    RecipeDef(
        "B_JUNK",
        frozenset({"B_junk"}),
        severity="warn",
        alpha_hint=0.75,
        priority=15,
        caa_tags=("B_JUNK",),
    ),
    RecipeDef(
        "C_OSC",
        frozenset({"C_osc"}),
        severity="tip",
        alpha_hint=0.5,
        priority=10,
        caa_tags=("C_OSC",),
    ),
    RecipeDef(
        "D_FOCUS",
        frozenset({"D_break"}),
        severity="warn",
        alpha_hint=0.5,
        priority=20,
        caa_tags=("D_FOCUS",),
    ),
    RecipeDef(
        "E_CLIFF",
        frozenset({"E_cliff"}),
        severity="tip",
        alpha_hint=0.5,
        priority=10,
        caa_tags=("E_CLIFF",),
    ),
    RecipeDef(
        "F_LENS",
        frozenset({"F_disagree"}),
        severity="tip",
        alpha_hint=0.5,
        priority=5,
        caa_tags=("F_LENS",),
    ),
    RecipeDef(
        "G_OOB",
        frozenset({"G_oob"}),
        severity="tip",
        alpha_hint=0.5,
        priority=5,
        caa_tags=("G_OOB",),
    ),
)


@dataclass
class RecipeMatch:
    recipe_id: str
    severity: Severity
    alpha_hint: float
    layer_hint: list[int]
    caa_tags: tuple[str, ...]
    priority: int
    atoms_mask: int


def atoms_from_flags(
    flags: list[str],
    *,
    f_disagree: bool = False,
    e_cliff: bool = False,
    g_oob: bool = False,
) -> dict[str, int]:
    """Map fine flags + async probes to atomic 0/1 dict."""
    atoms = {k: 0 for k in ATOM_KEYS}
    flag_set = set(flags)
    if "A_high_entropy" in flag_set or "A_low_margin" in flag_set:
        atoms["A_uncertain"] = 1
    if "B_distress" in flag_set:
        atoms["B_distress"] = 1
    if "B_stub" in flag_set:
        atoms["B_stub"] = 1
    if "B_junk" in flag_set:
        atoms["B_junk"] = 1
    if "C_oscillation" in flag_set or "C_thrash" in flag_set:
        atoms["C_osc"] = 1
    if "D_focus_break" in flag_set:
        atoms["D_break"] = 1
    if e_cliff:
        atoms["E_cliff"] = 1
    if f_disagree:
        atoms["F_disagree"] = 1
    if g_oob:
        atoms["G_oob"] = 1
    return atoms


def atoms_to_mask(atoms: dict[str, int]) -> int:
    mask = 0
    for name, bit in ATOM_BITS.items():
        if atoms.get(name):
            mask |= bit
    return mask


def _active_atoms(atoms: dict[str, int]) -> set[str]:
    return {k for k, v in atoms.items() if v}


def _recipe_matches(defn: RecipeDef, active: set[str]) -> bool:
    if not defn.required.issubset(active):
        return False
    if defn.any_b and not (active & _b_atoms()):
        return False
    return True


def match_recipe(atoms: dict[str, int], *, cliff_layer: int | None = None) -> RecipeMatch | None:
    """Return highest-priority matching recipe or None."""
    active = _active_atoms(atoms)
    if not active:
        return None
    best: RecipeMatch | None = None
    for defn in RECIPE_TABLE:
        if not _recipe_matches(defn, active):
            continue
        layer_hint = [cliff_layer if cliff_layer is not None else defn.layer_default]
        if cliff_layer is not None and cliff_layer not in layer_hint:
            layer_hint = [cliff_layer, defn.layer_default]
        cand = RecipeMatch(
            recipe_id=defn.recipe_id,
            severity=defn.severity,
            alpha_hint=defn.alpha_hint,
            layer_hint=layer_hint,
            caa_tags=defn.caa_tags,
            priority=defn.priority,
            atoms_mask=atoms_to_mask(atoms),
        )
        if best is None or cand.priority > best.priority:
            best = cand
    return best


_SEVERITY_RANK = {"none": 0, "tip": 1, "warn": 2, "critical": 3}


@dataclass
class RecipeMatcher:
    """Stateful matcher with debounce for live token stream."""

    debounce_tokens: int = DEBOUNCE_TOKENS
    _last_recipe: str | None = field(default=None, init=False)
    _last_fire_step: int = field(default=-10_000, init=False)
    _clear_streak: int = field(default=0, init=False)
    _eval_us_samples: list[float] = field(default_factory=list, init=False)

    def evaluate(
        self,
        step: int,
        atoms: dict[str, int],
        *,
        cliff_layer: int | None = None,
    ) -> RecipeMatch | None:
        t0 = time.perf_counter_ns()
        raw = match_recipe(atoms, cliff_layer=cliff_layer)
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        self._eval_us_samples.append(elapsed_us)
        if len(self._eval_us_samples) > 512:
            self._eval_us_samples = self._eval_us_samples[-256:]

        if raw is None:
            if self._last_recipe is not None:
                self._clear_streak += 1
                if self._clear_streak >= self.debounce_tokens:
                    self._last_recipe = None
            return None

        last_priority = 0
        if self._last_recipe is not None:
            last_priority = next(
                (d.priority for d in RECIPE_TABLE if d.recipe_id == self._last_recipe),
                0,
            )
        escalate = (
            self._last_recipe == raw.recipe_id
            and _SEVERITY_RANK.get(raw.severity, 0)
            > _SEVERITY_RANK.get(
                next(
                    (d.severity for d in RECIPE_TABLE if d.recipe_id == self._last_recipe),
                    "none",
                ),
                0,
            )
        )
        priority_escalation = (
            self._last_recipe is not None
            and raw.recipe_id != self._last_recipe
            and raw.priority > last_priority
        )
        if (
            self._last_recipe == raw.recipe_id
            and (step - self._last_fire_step) < self.debounce_tokens
            and not escalate
            and not priority_escalation
        ):
            return None

        self._last_recipe = raw.recipe_id
        self._last_fire_step = step
        self._clear_streak = 0
        return raw

    def reset(self) -> None:
        self._last_recipe = None
        self._last_fire_step = -10_000
        self._clear_streak = 0
        self._eval_us_samples.clear()

    def eval_timing_summary(self) -> dict[str, float]:
        xs = sorted(self._eval_us_samples)
        if not xs:
            return {"n": 0, "p50_us": 0.0, "p95_us": 0.0, "max_us": 0.0}
        n = len(xs)
        return {
            "n": float(n),
            "p50_us": xs[n // 2],
            "p95_us": xs[int(n * 0.95)],
            "max_us": xs[-1],
        }


def f_disagree_from_payload(payload: dict[str, Any]) -> bool:
    """True if any probe layer reports low J-lens vs sampling-head overlap."""
    probes = payload.get("probes") or {}
    for pdata in probes.values():
        if not isinstance(pdata, dict):
            continue
        if f_alarm_from_metrics(pdata.get("F")):
            return True
    return False


def recipe_catalog() -> list[dict[str, Any]]:
    """Static catalog for GET /chaos/recipes."""
    return [
        {
            "recipe_id": d.recipe_id,
            "required_atoms": sorted(d.required),
            "any_b": d.any_b,
            "severity": d.severity,
            "alpha_hint": d.alpha_hint,
            "layer_default": d.layer_default,
            "priority": d.priority,
            "caa_tags": list(d.caa_tags),
        }
        for d in sorted(RECIPE_TABLE, key=lambda x: -x.priority)
    ]

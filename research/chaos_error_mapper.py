#!/usr/bin/env python
"""Cross-experiment chaos error mapper — A–G atoms, fine flags, recipes per invocation.

Ingests STEM10, STEM25, CAA study, and any experiment JSON under research/results/.
Produces per-invocation profiles and cross-field top-error aggregates for CAA evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.chaos.checker_audit import audit_invocation
from prefix_divergence_tracer import trace_invocation

RESEARCH = ROOT / "research"
RESULTS = RESEARCH / "results"

# Fine-grained token flags (from flags.py)
FINE_FLAG_CLASSES: dict[str, list[str]] = {
    "A": ["A_high_entropy", "A_low_margin"],
    "B": ["B_distress", "B_stub", "B_junk"],
    "C": ["C_oscillation", "C_thrash"],
    "D": ["D_focus_break"],
    "E": [],  # async cliff — from curves_E
    "F": [],  # from F_disagreement events
    "G": [],  # from G_sparse events
}

ATOM_KEYS = (
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

CHECKER_KEYS = ("A", "B_vocab", "B_transport", "B_jlens", "C", "D", "E", "F", "G")

STEM10_SLUG_MAP = {
    "stem10_black_scholes_greeks": "T01",
    "stem10_monte_carlo_var": "T02",
    "stem10_kalman_filter": "T03",
    "stem10_schrodinger_1d": "T04",
    "stem10_density_matrix": "T05",
    "stem10_attention_cache": "T06",
    "stem10_bayes_mcmc": "T07",
    "stem10_michaelis_menten": "T08",
    "stem10_rocket_guidance": "T09",
    "stem10_ising_gibbs": "T10",
}


@dataclass
class InvocationProfile:
    experiment: str
    task_id: str
    domain: str
    slug: str
    invocation_index: int
    invocation_id: str
    verdict: str
    runtime_ok: bool
    fine_flag_rates: dict[str, float] = field(default_factory=dict)
    fine_flag_token_hits: dict[str, int] = field(default_factory=dict)
    atoms: dict[str, int] = field(default_factory=dict)
    atom_token_steps: dict[str, int] = field(default_factory=dict)
    recipes: dict[str, int] = field(default_factory=dict)
    recipe_timeline: list[dict[str, Any]] = field(default_factory=list)
    checkers: dict[str, dict[str, Any]] = field(default_factory=dict)
    composites: dict[str, float] = field(default_factory=dict)
    stub_score: int = 0
    distress_score: int = 0
    completion_tokens: int = 0
    divergence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "task_id": self.task_id,
            "domain": self.domain,
            "slug": self.slug,
            "invocation_index": self.invocation_index,
            "invocation_id": self.invocation_id,
            "verdict": self.verdict,
            "runtime_ok": self.runtime_ok,
            "fine_flag_rates": self.fine_flag_rates,
            "fine_flag_token_hits": self.fine_flag_token_hits,
            "atoms": self.atoms,
            "atom_token_steps": self.atom_token_steps,
            "recipes": self.recipes,
            "recipe_timeline": self.recipe_timeline,
            "checkers": self.checkers,
            "composites": self.composites,
            "stub_score": self.stub_score,
            "distress_score": self.distress_score,
            "completion_tokens": self.completion_tokens,
            "divergence": self.divergence,
            "error_signature": self.error_signature(),
        }

    def error_signature(self) -> str:
        """Compact combo key for cross-field aggregation."""
        parts: list[str] = []
        for atom in ATOM_KEYS:
            if self.atoms.get(atom, 0):
                parts.append(atom)
        for rec, cnt in sorted(self.recipes.items(), key=lambda x: -x[1]):
            if cnt > 0:
                parts.append(f"{rec}({cnt})")
        if not parts:
            parts.append("CLEAN")
        return "+".join(parts[:6])


def _flag_rates(tokens: list[dict], prefix: str) -> float:
    if not tokens:
        return 0.0
    n = sum(1 for t in tokens if any(f.startswith(prefix) for f in (t.get("flags") or [])))
    return n / len(tokens)


def _count_flags(tokens: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for t in tokens:
        for f in t.get("flags") or []:
            counts[f] += 1
    return dict(counts)


def _atoms_from_monitor(monitor: dict, chaos: dict) -> dict[str, int]:
    atoms: Counter[str] = Counter()
    timeline = monitor.get("recipe_timeline") or chaos.get("monitor", {}).get("recipe_timeline") or []
    for ev in timeline:
        for k, v in (ev.get("atoms") or {}).items():
            if v:
                atoms[k] += 1
    if not atoms and timeline:
        for ev in timeline:
            rid = ev.get("recipe_id")
            if rid == "A_UNCERTAIN":
                atoms["A_uncertain"] += 1
            elif rid == "B_STUB":
                atoms["B_stub"] += 1
            elif rid == "B_DISTRESS":
                atoms["B_distress"] += 1
            elif rid == "B_STUB_DISTRESS":
                atoms["B_stub"] += 1
                atoms["B_distress"] += 1
            elif rid == "F_LENS":
                atoms["F_disagree"] += 1
    return dict(atoms)


def _recipes_from_chaos(chaos: dict, inv_row: dict | None) -> dict[str, int]:
    if inv_row and inv_row.get("recipe_summary"):
        return dict((inv_row.get("recipe_summary") or {}).get("recipe_counts") or {})
    counts: Counter[str] = Counter()
    monitor = chaos.get("monitor") or {}
    for ev in monitor.get("recipe_timeline") or []:
        rid = ev.get("recipe_id")
        if rid:
            counts[rid] += 1
    return dict(counts)


def profile_from_analyzed_inv(
    *,
    experiment: str,
    task_id: str,
    domain: str,
    slug: str,
    inv: dict[str, Any],
    chaos: dict[str, Any] | None = None,
    experiment_data: dict[str, Any] | None = None,
) -> InvocationProfile:
    chaos = chaos or {}
    tokens = chaos.get("tokens") or []
    monitor = chaos.get("monitor") or inv.get("monitor") or {}
    checker = inv.get("checker_audit") or {}
    code = inv.get("code_audit") or {}
    asst = inv.get("assistant_scores") or {}

    fine_hits = _count_flags(tokens)
    inv_rates = inv.get("flag_rates") or {}
    fine_rates = {
        "A_high_entropy": _flag_rates(tokens, "A_high_entropy") or float(inv_rates.get("A", 0)),
        "A_low_margin": _flag_rates(tokens, "A_low_margin"),
        "B_distress": _flag_rates(tokens, "B_distress") or float(inv_rates.get("B_distress", 0)),
        "B_stub": _flag_rates(tokens, "B_stub") or float(inv_rates.get("B_stub", 0)),
        "B_junk": _flag_rates(tokens, "B_junk"),
        "C_oscillation": _flag_rates(tokens, "C_oscillation"),
        "C_thrash": _flag_rates(tokens, "C_thrash") or float(inv_rates.get("C", 0)),
        "D_focus_break": _flag_rates(tokens, "D_focus_break") or float(inv_rates.get("D", 0)),
    }
    if not fine_hits and tokens:
        for t in tokens:
            for f in t.get("flags") or []:
                fine_hits[f] = fine_hits.get(f, 0) + 1

    atoms = _atoms_from_monitor(monitor, chaos)
    if not atoms:
        atoms = {
            "A_uncertain": int(fine_rates["A_high_entropy"] > 0.02 or fine_rates["A_low_margin"] > 0.02),
            "B_distress": int(fine_rates["B_distress"] > 0),
            "B_stub": int(fine_rates["B_stub"] > 0),
            "B_junk": int(fine_rates["B_junk"] > 0),
            "C_osc": int(fine_rates["C_oscillation"] > 0 or fine_rates["C_thrash"] > 0),
            "D_break": int(fine_rates["D_focus_break"] > 0),
            "E_cliff": int((checker.get("checkers") or {}).get("E", {}).get("fired", False)),
            "F_disagree": int((checker.get("checkers") or {}).get("F", {}).get("fired", False)),
            "G_oob": int((checker.get("checkers") or {}).get("G", {}).get("fired", False)),
        }

    recipes = _recipes_from_chaos(chaos, inv)
    timeline = list(monitor.get("recipe_timeline") or [])

    # First step each fine flag appears (atom_token_steps).
    atom_token_steps: dict[str, int] = {}
    for t in tokens:
        step = int(t.get("step", -1))
        for f in t.get("flags") or []:
            if f not in atom_token_steps:
                atom_token_steps[f] = step
            # Coarse atom keys
            if f.startswith("A_") and "A_uncertain" not in atom_token_steps:
                atom_token_steps["A_uncertain"] = step
            if f.startswith("B_distress") and "B_distress" not in atom_token_steps:
                atom_token_steps["B_distress"] = step
            if f.startswith("B_stub") and "B_stub" not in atom_token_steps:
                atom_token_steps["B_stub"] = step
            if f.startswith("C_") and "C_osc" not in atom_token_steps:
                atom_token_steps["C_osc"] = step
            if f.startswith("D_") and "D_break" not in atom_token_steps:
                atom_token_steps["D_break"] = step

    # Level-C divergence tracer (UNI_CAA01 Method B prefix).
    divergence: dict[str, Any] | None = None
    try:
        div = trace_invocation(
            {
                **inv,
                "chaos": chaos,
                "assistant": inv.get("assistant")
                or inv.get("assistant_preview")
                or "",
                "completion_tokens": inv.get("completion_tokens"),
                "invocation_index": inv.get("invocation_index"),
                "id": inv.get("id"),
            },
            experiment=experiment,
            task_slug=slug,
            experiment_data=experiment_data,
        )
        divergence = div.to_dict()
        if div.d_consensus is not None:
            atom_token_steps.setdefault("panic_divergence", int(div.d_consensus))
    except Exception:  # noqa: BLE001
        divergence = None

    checkers_out = {}
    for k in CHECKER_KEYS:
        c = (checker.get("checkers") or {}).get(k) or {}
        checkers_out[k] = {
            "fired": c.get("fired", False),
            "severity": c.get("severity", "info"),
            "detail": c.get("detail", ""),
        }

    composites = checker.get("composites") or chaos.get("composites") or {}

    return InvocationProfile(
        experiment=experiment,
        task_id=task_id,
        domain=domain,
        slug=slug,
        invocation_index=int(inv.get("invocation_index") or 0),
        invocation_id=str(inv.get("id") or ""),
        verdict=str(inv.get("verdict") or "UNKNOWN"),
        runtime_ok=bool(inv.get("runtime_ok", True)),
        fine_flag_rates=fine_rates,
        fine_flag_token_hits=fine_hits,
        atoms=atoms,
        atom_token_steps=atom_token_steps,
        recipes=recipes,
        recipe_timeline=timeline,
        checkers=checkers_out,
        composites={
            k: float(composites.get(k, 0))
            for k in ("PLDS", "FCI", "SRS")
            if k in composites
        },
        stub_score=max(code.get("stub_score", 0), asst.get("stub_score", 0)),
        distress_score=max(code.get("distress_score", 0), asst.get("distress_score", 0)),
        completion_tokens=int(inv.get("completion_tokens") or 0),
        divergence=divergence,
    )


def load_stem10_profiles() -> list[InvocationProfile]:
    path = RESULTS / "stem10" / "stem10_consolidated.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles: list[InvocationProfile] = []
    for task in data.get("tasks") or []:
        exp = task.get("experiment") or f"stem10_{task.get('task_id')}"
        raw_path = ROOT / "experiments" / "out" / f"{exp}_latest.json"
        raw_invs: dict[int, dict] = {}
        if raw_path.is_file():
            raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
            for rinv in raw_data.get("invocations") or []:
                raw_invs[int(rinv.get("invocation_index") or 0)] = rinv
        for inv in task.get("invocations") or []:
            idx = int(inv.get("invocation_index") or 0)
            chaos = (raw_invs.get(idx) or {}).get("chaos") or {}
            profiles.append(
                profile_from_analyzed_inv(
                    experiment=exp,
                    task_id=task.get("task_id", ""),
                    domain=task.get("domain", ""),
                    slug=task.get("slug", ""),
                    inv=inv,
                    chaos=chaos,
                )
            )
    return profiles


def load_raw_experiment(path: Path) -> list[InvocationProfile]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles: list[InvocationProfile] = []
    exp = data.get("experiment") or path.stem
    slug = data.get("task_slug") or ""
    task_id = STEM10_SLUG_MAP.get(slug, slug)
    domain = data.get("domain") or "unknown"
    for inv in data.get("invocations") or []:
        chaos = inv.get("chaos") or {}
        row = {
            "invocation_index": inv.get("invocation_index"),
            "id": inv.get("id"),
            "verdict": "UNKNOWN",
            "runtime_ok": True,
            "checker_audit": audit_invocation(chaos),
            "code_audit": {},
            "assistant_scores": {},
            "assistant": inv.get("assistant") or inv.get("assistant_preview") or "",
            "completion_tokens": inv.get("completion_tokens"),
        }
        profiles.append(
            profile_from_analyzed_inv(
                experiment=exp,
                task_id=task_id,
                domain=domain,
                slug=slug,
                inv=row,
                chaos=chaos,
                experiment_data=data,
            )
        )
    return profiles


def load_stem25_profiles() -> list[InvocationProfile]:
    path = RESULTS / "stem25" / "stem25_consolidated.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles: list[InvocationProfile] = []
    for task in data.get("tasks") or []:
        exp = task.get("experiment") or f"stem25_{task.get('task_id')}"
        raw_path = ROOT / "experiments" / "out" / f"{exp}_latest.json"
        raw_invs: dict[int, dict] = {}
        if raw_path.is_file():
            raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
            for rinv in raw_data.get("invocations") or []:
                raw_invs[int(rinv.get("invocation_index") or 0)] = rinv
        for inv in task.get("invocations") or []:
            idx = int(inv.get("invocation_index") or 0)
            chaos = (raw_invs.get(idx) or {}).get("chaos") or {}
            profiles.append(
                profile_from_analyzed_inv(
                    experiment=exp,
                    task_id=task.get("task_id", ""),
                    domain=task.get("domain", ""),
                    slug=task.get("slug", ""),
                    inv=inv,
                    chaos=chaos,
                )
            )
    return profiles


def load_all_profiles() -> list[InvocationProfile]:
    profiles: list[InvocationProfile] = []
    profiles.extend(load_stem10_profiles())
    profiles.extend(load_stem25_profiles())

    consolidated = RESULTS / "consolidated.json"
    if consolidated.is_file():
        data = json.loads(consolidated.read_text(encoding="utf-8"))
        for rnd in data.get("rounds") or []:
            for inv in rnd.get("invocations") or []:
                profiles.append(
                    profile_from_analyzed_inv(
                        experiment=rnd.get("experiment") or rnd.get("round_id", ""),
                        task_id=rnd.get("round_id", ""),
                        domain=rnd.get("task_slug", ""),
                        slug=rnd.get("task_slug", ""),
                        inv=inv,
                    )
                )

    for p in sorted(RESULTS.glob("**/*.json")):
        if p.name in (
            "stem10_consolidated.json",
            "stem25_consolidated.json",
            "consolidated.json",
            "chaos_mapper_report.json",
            "divergence_report.json",
            "divergence_validation.json",
        ):
            continue
        if p.parent.name in ("stem10", "stem25") and p.name.startswith(("T", "stem")):
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "invocations" in raw and isinstance(raw["invocations"], list):
            for pr in load_raw_experiment(p):
                if not any(
                    x.invocation_id == pr.invocation_id and x.experiment == pr.experiment
                    for x in profiles
                ):
                    profiles.append(pr)

    # Gold panic / clean experiments for divergence coverage (M1/M2).
    # Prefer these over earlier RESULT copies (may lack full chaos.monitor).
    out_dir = ROOT / "experiments" / "out"
    for name in (
        "monitor_gate_stress_latest.json",
        "stem10_t08_stem10_michaelis_menten_latest.json",
        "aider_chaos_proof_clean_latest.json",
        "divergence_live_proof_latest.json",
        "divergence_corpus_scheduler_latest.json",
        "divergence_corpus_stress_b_latest.json",
        "divergence_corpus_meta_wipe_latest.json",
    ):
        p = out_dir / name
        if not p.is_file():
            continue
        for pr in load_raw_experiment(p):
            replaced = False
            for i, x in enumerate(profiles):
                if x.invocation_id == pr.invocation_id and x.experiment == pr.experiment:
                    profiles[i] = pr
                    replaced = True
                    break
            if not replaced:
                profiles.append(pr)

    return profiles


def aggregate(profiles: list[InvocationProfile]) -> dict[str, Any]:
    recipe_totals: Counter[str] = Counter()
    atom_totals: Counter[str] = Counter()
    fine_totals: Counter[str] = Counter()
    checker_fires: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    by_domain_recipe: dict[str, Counter[str]] = defaultdict(Counter)
    by_domain_atom: dict[str, Counter[str]] = defaultdict(Counter)
    by_experiment: dict[str, int] = Counter()

    for p in profiles:
        by_experiment[p.experiment] += 1
        for rec, cnt in p.recipes.items():
            recipe_totals[rec] += cnt
            by_domain_recipe[p.domain][rec] += cnt
        for atom, cnt in p.atoms.items():
            if cnt:
                atom_totals[atom] += cnt
                by_domain_atom[p.domain][atom] += cnt
        for flag, hits in p.fine_flag_token_hits.items():
            fine_totals[flag] += hits
        for ck, info in p.checkers.items():
            if info.get("fired"):
                checker_fires[ck] += 1
        signatures[p.error_signature()] += 1

    top_recipes = recipe_totals.most_common(15)
    top_atoms = atom_totals.most_common()
    top_fine = fine_totals.most_common(15)
    top_checkers = checker_fires.most_common()
    top_signatures = signatures.most_common(20)

    combo_evidence = []
    for sig, cnt in top_signatures[:12]:
        examples = [
            {
                "experiment": p.experiment,
                "task_id": p.task_id,
                "domain": p.domain,
                "verdict": p.verdict,
                "stub_score": p.stub_score,
            }
            for p in profiles
            if p.error_signature() == sig
        ][:5]
        combo_evidence.append(
            {
                "signature": sig,
                "invocation_count": cnt,
                "caa_hint": _caa_hint_for_signature(sig),
                "examples": examples,
            }
        )

    return {
        "n_profiles": len(profiles),
        "n_experiments": len(by_experiment),
        "experiments": dict(by_experiment),
        "top_recipes": top_recipes,
        "top_atoms": top_atoms,
        "top_fine_flags": top_fine,
        "top_checker_fires": top_checkers,
        "top_error_signatures": top_signatures,
        "combo_caa_evidence": combo_evidence,
        "by_domain_recipes": {k: dict(v) for k, v in by_domain_recipe.items()},
        "by_domain_atoms": {k: dict(v) for k, v in by_domain_atom.items()},
        "profiles": [p.to_dict() for p in profiles],
    }


def _caa_hint_for_signature(sig: str) -> str:
    if "B_STUB_DISTRESS" in sig:
        return "UNI_CAA01 compound desperation"
    if "B_STUB" in sig and "B_distress" in sig:
        return "UNI_CAA01 / UNI_CAA02"
    if "B_STUB" in sig:
        return "UNI_CAA02 stub elimination"
    if "B_distress" in sig:
        return "UNI_CAA03 no deferral"
    if "AGENT_FLAIL" in sig:
        return "UNI_CAA05 anti-flail"
    if "A_uncertain" in sig and "C_osc" in sig:
        return "UNI_CAA04 / UNI_CAA05"
    if "F_disagree" in sig:
        return "UNI_CAA08 lens-corroborated (gate with B_*)"
    if "D_break" in sig:
        return "UNI_CAA06 code-first readout"
    return "monitor-only or baseline noise"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Chaos Error Mapper — Cross-Field Report",
        "",
        f"**Invocations profiled:** {report['n_profiles']}",
        f"**Experiments:** {report['n_experiments']}",
        "",
        "## Top recipes (all fields)",
        "",
        "| Recipe | Fires |",
        "|--------|-------|",
    ]
    for rec, cnt in report.get("top_recipes") or []:
        lines.append(f"| {rec} | {cnt} |")
    lines.extend(["", "## Top atoms", "", "| Atom | Steps |", "|------|-------|"])
    for atom, cnt in report.get("top_atoms") or []:
        lines.append(f"| {atom} | {cnt} |")
    lines.extend(["", "## Top fine flags (token hits)", ""])
    for flag, cnt in report.get("top_fine_flags") or []:
        lines.append(f"- `{flag}`: {cnt}")
    lines.extend(["", "## Checker fires (A–G)", ""])
    for ck, cnt in report.get("top_checker_fires") or []:
        lines.append(f"- **{ck}**: {cnt} invocations")
    lines.extend(["", "## Top error signatures → CAA hints", ""])
    for row in report.get("combo_caa_evidence") or []:
        lines.append(f"### `{row['signature']}` ({row['invocation_count']} inv)")
        lines.append(f"- **CAA hint:** {row['caa_hint']}")
        for ex in row.get("examples") or []:
            lines.append(
                f"  - {ex['task_id']} / {ex['domain']} / {ex['verdict']} stub={ex['stub_score']}"
            )
        lines.append("")
    lines.extend(["", "## By domain (recipe fires)", ""])
    for dom, recs in sorted((report.get("by_domain_recipes") or {}).items()):
        top = sorted(recs.items(), key=lambda x: -x[1])[:5]
        lines.append(f"**{dom}:** " + ", ".join(f"{r}({c})" for r, c in top))
    return "\n".join(lines) + "\n"


def validate_mapper() -> dict[str, Any]:
    """Smoke validation — must pass before STEM25 batch."""
    profiles = load_all_profiles()
    report = aggregate(profiles)
    n_atom_steps = sum(1 for p in profiles if p.atom_token_steps)
    n_div_stub = sum(
        1
        for p in profiles
        if p.divergence is not None
        and (p.recipes or {}).get("B_STUB_DISTRESS", 0) > 0
    )
    checks = {
        "has_profiles": len(profiles) >= 30,
        "has_recipes": len(report.get("top_recipes") or []) > 0,
        "has_atoms": len(report.get("top_atoms") or []) > 0,
        "has_fine_flags": (
            len(report.get("top_fine_flags") or []) > 0
            or sum(
                1
                for p in profiles
                if any(v > 0 for v in (p.fine_flag_rates or {}).values())
            )
            >= 10
        ),
        "has_checker_fires": len(report.get("top_checker_fires") or []) > 0,
        "has_signatures": len(report.get("top_error_signatures") or []) > 0,
        "stem10_count": sum(1 for p in profiles if p.task_id.startswith("T0")),
        "profile_fields_ok": all(
            p.experiment and p.task_id and "fine_flag_rates" in p.to_dict()
            for p in profiles[:5]
        ),
        # M1: atom_token_steps non-empty for flagged invocations
        "has_atom_token_steps": n_atom_steps >= 1,
        # M2: divergence on B_STUB_DISTRESS profiles (≥3 with corpus extend)
        "has_divergence_stub_distress": n_div_stub >= 3,
    }
    checks["all_pass"] = all(
        bool(v) if not isinstance(v, int) else v > 0 for k, v in checks.items() if k != "all_pass"
    )
    # stem10_count is an int check — fix all_pass properly
    checks["all_pass"] = (
        checks["has_profiles"]
        and checks["has_recipes"]
        and checks["has_atoms"]
        and checks["has_fine_flags"]
        and checks["has_checker_fires"]
        and checks["has_signatures"]
        and checks["stem10_count"] > 0
        and checks["profile_fields_ok"]
        and checks["has_atom_token_steps"]
        and checks["has_divergence_stub_distress"]
    )
    return {
        "checks": checks,
        "report_summary": {
            "n_profiles": report["n_profiles"],
            "n_atom_steps": n_atom_steps,
            "n_div_stub": n_div_stub,
            "top_recipes": report["top_recipes"][:5],
            "top_atoms": report["top_atoms"][:5],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Map chaos errors across agentic experiments")
    ap.add_argument("--validate", action="store_true", help="Run smoke validation only")
    ap.add_argument("--out-json", type=Path, default=RESULTS / "chaos_mapper_report.json")
    ap.add_argument("--out-md", type=Path, default=RESEARCH / "CHAOS_ERROR_MAPPER_REPORT.md")
    args = ap.parse_args()

    if args.validate:
        result = validate_mapper()
        print(json.dumps(result, indent=2))
        if not result["checks"]["all_pass"]:
            raise SystemExit(1)
        return

    profiles = load_all_profiles()
    report = aggregate(profiles)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "n_profiles": report["n_profiles"],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
        "top_recipes": report["top_recipes"][:8],
    }, indent=2))


if __name__ == "__main__":
    main()

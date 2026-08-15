"""Golden bitmask tests for RecipeMatcher."""

from __future__ import annotations

from jlens_qwen.chaos.recipes import (
    ATOM_BITS,
    RecipeMatcher,
    atoms_from_flags,
    atoms_to_mask,
    match_recipe,
)


def test_atoms_from_flags_distress_stub():
    atoms = atoms_from_flags(["B_distress", "B_stub"])
    assert atoms["B_distress"] == 1
    assert atoms["B_stub"] == 1
    assert atoms["A_uncertain"] == 0


def test_match_b_stub_distress_combo():
    atoms = atoms_from_flags(["B_distress", "B_stub"])
    m = match_recipe(atoms)
    assert m is not None
    assert m.recipe_id == "B_STUB_DISTRESS"
    assert m.severity == "warn"
    assert m.alpha_hint == 0.75


def test_match_panic_full_priority():
    atoms = atoms_from_flags(
        ["A_low_margin", "B_distress", "C_thrash"],
        f_disagree=False,
    )
    m = match_recipe(atoms)
    assert m is not None
    assert m.recipe_id == "PANIC_FULL"
    assert m.severity == "critical"


def test_match_f_lens_tip_only():
    atoms = atoms_from_flags([], f_disagree=True)
    m = match_recipe(atoms)
    assert m is not None
    assert m.recipe_id == "F_LENS"
    assert m.severity == "tip"


def test_match_agent_flail():
    atoms = atoms_from_flags(["B_stub", "B_distress"], f_disagree=True)
    m = match_recipe(atoms)
    assert m is not None
    assert m.recipe_id == "AGENT_FLAIL"


def test_no_match_calm():
    atoms = atoms_from_flags([])
    assert match_recipe(atoms) is None


def test_atoms_mask_golden():
    atoms = {
        "A_uncertain": 1,
        "B_distress": 0,
        "B_stub": 1,
        "B_junk": 0,
        "C_osc": 0,
        "D_break": 0,
        "E_cliff": 0,
        "F_disagree": 0,
        "G_oob": 0,
    }
    mask = atoms_to_mask(atoms)
    assert mask == (ATOM_BITS["A_uncertain"] | ATOM_BITS["B_stub"])


def test_debounce_same_recipe():
    matcher = RecipeMatcher(debounce_tokens=4)
    atoms = atoms_from_flags(["B_distress", "B_stub"])
    m0 = matcher.evaluate(0, atoms)
    m1 = matcher.evaluate(1, atoms)
    assert m0 is not None
    assert m1 is None
    m4 = matcher.evaluate(4, atoms)
    assert m4 is not None


def test_debounce_priority_escalation_bypasses():
    matcher = RecipeMatcher(debounce_tokens=8)
    low = atoms_from_flags(["B_distress", "B_stub"])
    high = atoms_from_flags(
        ["A_low_margin", "B_distress", "C_thrash"],
    )
    m0 = matcher.evaluate(0, low)
    assert m0 is not None
    assert m0.recipe_id == "B_STUB_DISTRESS"
    m1 = matcher.evaluate(1, high)
    assert m1 is not None
    assert m1.recipe_id == "PANIC_FULL"


def test_e_cliff_alarm_requires_magnitude():
    from jlens_qwen.chaos.flags import e_cliff_alarm

    assert not e_cliff_alarm(16, 0.1)
    assert e_cliff_alarm(16, 0.5)
    assert not e_cliff_alarm(5, 0.9)


def test_recipe_eval_speed_budget():
    """Recipe evaluate must stay ≤50µs p95 on CPU (10k iterations)."""
    import time

    matcher = RecipeMatcher()
    atoms = atoms_from_flags(["B_distress", "B_stub", "C_thrash"], f_disagree=True)
    samples: list[float] = []
    for step in range(10_000):
        t0 = time.perf_counter_ns()
        matcher.evaluate(step % 16, atoms, cliff_layer=16)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    samples.sort()
    p95 = samples[int(len(samples) * 0.95)]
    assert p95 <= 50.0, f"recipe p95={p95:.2f}µs exceeds 50µs budget"

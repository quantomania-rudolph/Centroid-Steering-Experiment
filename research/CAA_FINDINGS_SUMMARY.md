# CAA Study — Executive Brief

**Date:** 2026-08-06  
**Artifacts:** `research/results/consolidated.json`, `research/CAA_STEERING_REPORT.md`

## Method

- **5 planned rounds** (clean, stub, distress, combo scheduler, topo trajectory)
- **R1–R3 succeeded** (7 invocations, all `RESOLVED`, lint+runtime pass)
- **R4–R5 blocked** by Qwen server `InternalServerError` during long session (0 LLM calls logged)
- **Historical merge:** 3 prior scheduler stress runs (`aider_calib_stress_v2`, `monitor_gate_stress`, `live_panic_demo_scheduler`) — 8 additional invocations

**Cross-reference per invocation:**
- A–G `recipe_timeline` + per-token flag rates (first vs last quartile)
- Final `main.py` stub/distress grep
- Ruff/pylint + runtime assert

## Critical findings

### Errors that self-resolve (monitor only — skip CAA)

| Signal | Evidence | Why skip CAA |
|--------|----------|--------------|
| `F_LENS` | 63 historical fires; fires on clean fibonacci | Code correct; J-lens overlap=0 is baseline on Qwen |
| `A_UNCERTAIN` | Tips on first tokens | Settles; no bad code |
| `B_distress` alone | R3 distress bucket: 0% flag rate when model commits | Resolves once patch starts |

### Errors that fire during panic but may still end OK

| Signal | Evidence | CAA role |
|--------|----------|----------|
| `B_STUB_DISTRESS` | H2 inv#1: fired 2×; B_stub 2.3% tokens; **did not clear** first→last quartile | Gate CAA when recipe fires **and** stub_score>0 in output |
| `AGENT_FLAIL` | H2 inv#1: 1 fire with triple combo | Escalate α; same gate as above |
| `B_STUB` | H2 inv#1: 2 fires; stub mass in early gen | Use when `NotImplementedError` in assistant stream |

**Key insight:** Chaos flags detect **in-flight distress** better than final code quality. CAST-style gating (Lee et al. 2024) must combine `recipe_id` with **artifact check** (stub grep or failed runtime) to avoid steering clean completions.

### Errors we could not link to unresolved stubs in this study

All 15 analyzed invocations ended with `stub_score=0` and passing lint. Harder tasks (`expr_compiler`) need a follow-up round when server is stable.

## Recommended CAA vectors (8) — ranked

| Priority | CAA ID | Monitor gate | α | Layer |
|----------|--------|--------------|---|-------|
| **HIGH** | CAA_01_B_STUB_DISTRESS | `B_stub ∧ B_distress` | 0.75 | L16 |
| **MEDIUM** | CAA_02_AGENT_FLAIL | `B_stub ∧ B_distress ∧ F_disagree` | 1.0 | L16 |
| **MEDIUM** | CAA_03_B_STUB_ONLY | `B_stub` warn+ | 0.75 | L16 |
| LOW | CAA_04_B_DISTRESS_ONLY | `B_distress` | 0.75 | L16 |
| LOW | CAA_05_PANIC_FULL | `A ∧ B_* ∧ C_osc` | 1.0 | L16 |
| LOW | CAA_06_A_C_VACILLATE | `A ∧ C_osc` | 0.5 | L16 |
| LOW | CAA_07_B_F_SURFACE | `B_* ∧ F` | 0.75 | L16, L10 |
| LOW | CAA_08_D_F_READOUT | `D ∧ F` (gen_start only) | 0.5 | L16 |

Full pos/neg prompt pairs: see `research/CAA_STEERING_REPORT.md`.

## Research anchors

1. **Rimsky et al. 2024** — mean activation difference; ACL Outstanding Paper  
2. **Lee et al. 2024 (CAST)** — conditional steering beats unconditional  
3. **Zou et al. 2023 (RepE)** — behavioral opposites, not topic pairs  
4. **Rahman et al. 2026** — code-style directions on Qwen2.5-Coder  

## Next steps

1. Re-run R4/R5 when server stable (combo + trajectory tasks)
2. Add `expr_compiler` stress round for truly unresolved stub artifacts
3. Build CAA dataset from pos/neg pairs above at L10/L16/L22
4. Validate CAST gate: `recipe_id != NONE AND stub_score > 0`

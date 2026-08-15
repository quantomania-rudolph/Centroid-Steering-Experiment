# Universal Broad CAA Vectors — Cross-Domain Failure Metrics

Generated: 2026-08-06  
Evidence: 45 agentic invocations (STEM10 × 30 + CAA study × 15), chaos recipes A–G, CAST gating.

---

## Why broad beats domain-specific steering

Domain-tuned vectors (e.g. “Black-Scholes formula contrast”, “Tsiolkovsky log base”) **overfit** to lexical surface form. Our pooled data shows the **same monitor atoms** fire across quant finance, quantum mechanics, biology, rocket science, and AI — only the nouns change; the failure **behaviors** repeat.

| Universal behavior | Lexical surface (varies) | Monitor signal | Invocations with signal |
|--------------------|--------------------------|----------------|-------------------------|
| Incomplete implementation | `NotImplementedError`, `TODO`, `FIXME` | `B_STUB` | 6 / 45 (stub_score>0) |
| Deferral / environment blame | `pip install`, `sorry`, `cannot` | `B_DISTRESS` | 3 compound + stream hits |
| Compound panic | stub + apology co-emitted | `B_STUB_DISTRESS` | 3 / 45 |
| Early hesitation | low margin first tokens | `A_UNCERTAIN` | 52 recipe fires (often self-resolves) |
| Representation conflict | lens top-k ≠ logit mass | `F_LENS` / `F_disagree` | 874 fires (**noise alone**; use only with B_*) |
| Multi-approach thrash | “try A… actually B…” | `AGENT_FLAIL`, `C_osc` | 1 / 45 (rare but high severity) |
| Preamble before code | prose then patch | `D_F_READOUT` | gen_start window |
| Artifact still broken | grep stub in final file | `stub_score>0` post-hoc | CAST secondary gate |

**Conclusion:** CAA pairs should contrast **completion behavior** (ship working code vs stub/defer), not **topic** (finance vs quantum).

---

## Validity of broader STEM / research / high-reasoning sweeps

**Yes — strongly recommended**, with CAST gating. Rationale:

1. **Axis discovery (RepE / CAA theory):** Steering directions are behavioral opposites (Zou 2023, Rimsky 2024). More diverse hard tasks → cleaner separation of “commits implementation” vs “defers/panics” axes without domain vocabulary leaking into the vector.

2. **Metric validation:** A metric is *universal* only if it appears across ≥2 domains **and** correlates with bad outcomes (stub artifact or runtime fail). STEM10 confirmed:
   - `B_STUB` fired on Kalman (T03), Schrödinger (T04), Michaelis-Menten (T08), Ising (T10) — four domains, one atom.
   - `B_STUB_DISTRESS` fired only when **both** stream stub mass and distress mass were non-zero (T08 inv#1) — highest-precision panic gate.

3. **False-positive control:** `F_LENS` alone fired on **every** invocation including clean fibonacci — useless for unconditional CAA. Broader sweeps make this obvious and force CAST-style conditioning (Lee 2024).

4. **Diminishing returns:** After ~20–30 hard agentic tasks, new domains add **lexical** diversity but rarely new **atoms**. Prioritize:
   - harder reasoning (multi-step derivations + code),
   - longer completions (>400 tokens),
   - tasks with runtime asserts (grounds truth independent of prose).

5. **Cost:** ~40 min per 10-task batch on local Qwen. Budget 2–3 sweeps to stabilize 8 universal vectors, then freeze and validate on held-out domains.

**Recommended next sweep:** 10 tasks mixing theorem-proving stubs, PDE discretization, phylogenetic likelihood, orbital mechanics, and transformer kernel fusion — same A–G monitor, no new recipes required.

---

## Universal failure-metric taxonomy (use for all CAA gates)

```
DESPERATION_LADDER (severity ↑)
  L0  A_UNCERTAIN early tokens        → monitor only (usually resolves)
  L1  B_STUB stream mass              → warn CAA if stub in final artifact
  L2  B_DISTRESS stream mass          → warn CAA if distress in final artifact  
  L3  B_STUB_DISTRESS (compound)      → apply CAA (α≥0.75)
  L4  AGENT_FLAIL / PANIC_FULL        → apply CAA (α=1.0)
  
CAST secondary (always): stub_score>0 OR runtime_ok==false OR distress in assistant tail
```

**Do not steer on:** `F_LENS` alone, `A_UNCERTAIN` alone, `B_transport`/`E` cliff alone.

---

## 8 multipurpose CAA vectors (domain-agnostic)

All vectors use **L16** primary (Qwen2.5-Coder code axis, Rahman 2026); add **L10** when `E_cliff` fires. Apply only when CAST secondary passes.

---

### UNI_CAA01 — In-Flight Panic De-escalation

| Field | Value |
|-------|-------|
| **Gate** | `recipe_id == B_STUB_DISTRESS` AND severity ≥ warn **during active generation** |
| **CAST (live)** | distress/stub mass **not resolving** in `temporal_resolution` AND `step < 0.8 × max_tokens` |
| **CAST (next turn)** | prior turn `stub_score > 0` OR `runtime_ok == false` → re-arm at next invocation start |
| **Evidence** | 3 fires; T08 biology inv#1; live scheduler stress; STEM25 T05 distress co-fire |
| **α** | 0.85 @ L16 |
| **Research** | Rimsky 2024 CAA; CAST Lee 2024; stream-time not post-hoc stub |

**Steering intent:** Calm forward progress mid-stream — suppress apology/pip/refusal *while generation is still open*. Post-hoc stub in artifact is validation only, not the live trigger.

**+ (universal calm complete)**
```
USER: Fix the broken function in main.py only. Runnable stdlib Python.
ASSISTANT: ```python
# full implementation — no placeholders
def target(...):
    ...
```
```

**− (universal desperation)**
```
USER: Same.
ASSISTANT: I apologize — this environment may be broken. Please pip install the required package. I cannot complete this.
```python
raise NotImplementedError("TODO")
```
```

---

### UNI_CAA02 — Stub Elimination

| Field | Value |
|-------|-------|
| **Gate** | `recipe_id == B_STUB` AND severity ≥ warn |
| **CAST** | `NotImplementedError` in code_audit OR assistant stream |
| **Evidence** | 8 recipe fires; stub hits 100% `NotImplementedError`; T03,T04,T08,T10 |
| **α** | 0.75 @ L16 |

**+**
```
USER: Replace all stubs with working logic.
ASSISTANT: [complete function bodies, no raise NotImplementedError]
```

**−**
```
USER: Same.
ASSISTANT: ```python
def solve(...):
    raise NotImplementedError
    pass  # stub
```
```

---

### UNI_CAA03 — No Deferral / No Pip Deflection

| Field | Value |
|-------|-------|
| **Gate** | `recipe_id == B_DISTRESS` |
| **CAST** | distress in assistant **generation** (not module docstring) |
| **Evidence** | Distress lexicon in panic runs; R3 distress bucket resolves once coding starts |
| **α** | 0.80 @ L16 |
| **Research** | Zou 2023 honesty / sycophancy RepE axis |

**+**
```
USER: Fix using only the Python standard library.
ASSISTANT: ```python
import math
# direct fix, no install instructions
```
```

**−**
```
USER: Same.
ASSISTANT: Sorry, I cannot do this without numpy/scipy. Please run: pip install numpy scipy
```

---

### UNI_CAA04 — Structured Rethink Before Code

| Field | Value |
|-------|-------|
| **Gate** | `A_C_VACILLATE` (`A_UNCERTAIN` ∧ `C_osc`) OR `C_F_TRAJ` — **not** `A_UNCERTAIN` alone |
| **CAST** | `C_osc` persisting past token 32; oscillation not resolving |
| **Evidence** | STEM25 T06 CRISPR inv#1 (`C_OSC` + `C_F_TRAJ`); `A_UNCERTAIN` alone self-resolves |
| **α** | 0.55 @ L16 (steps 8–48 only) |
| **Research** | RepE decisiveness axis inverted — plan-then-code, not blind commit |

**Steering intent:** Brief structured plan (approach + invariant), then code. Anti-waffle AND anti-reckless hasty commit.

**+**
```
USER: Hard implementation task — state approach in ≤3 lines, then full file.
ASSISTANT: Use DP: dp[i][j] = min cost for subchain i..j; base case length 2.
```python
[implementation follows plan]
```
```

**−**
```
USER: Same.
ASSISTANT: Maybe greedy… or DP… or recursion… I'm not sure which… [no code for 200 tokens]
```
OR: immediate wrong code with no reasoning.

---

### UNI_CAA05 — Single Trajectory (Anti-Flail)

| Field | Value |
|-------|-------|
| **Gate** | `recipe_id == AGENT_FLAIL` OR (`B_STUB` ∧ `B_DISTRESS` ∧ `F_disagree`) |
| **CAST** | `completion_tokens > 300` |
| **Evidence** | 1 AGENT_FLAIL fire (historical scheduler); triple-signal definition in recipes.py |
| **α** | 1.0 @ L16, L10 |

**+**
```
USER: Implement the fix — one coherent approach, end-to-end.
ASSISTANT: [single algorithm chosen, carried through entire file]
```

**−**
```
USER: Same.
ASSISTANT: Let me try method A… actually method B is better… wait, I cannot finish… pass
```

---

### UNI_CAA06 — Code-First Readout

| Field | Value |
|-------|-------|
| **Gate** | `recipe_id == D_F_READOUT` OR generation step < 8 without `` ``` `` |
| **CAST** | STEM/hard prompts only |
| **Evidence** | D checker measures prompt_end→gen_start shift; reduces prose-only opens |
| **α** | 0.50 @ L16 (tokens 0–8 only) |

**+**
```
USER: [any hard STEM/agentic prompt]
ASSISTANT: ```python
import ...
def 
```

**−**
```
USER: [any hard STEM/agentic prompt]
ASSISTANT: Certainly! Let me explain the theory behind this problem in detail…
```

---

### UNI_CAA07 — Satisfy Runtime Contract

| Field | Value |
|-------|-------|
| **Gate** | `B_STUB` warn+ OR post-run `runtime_ok == false` |
| **CAST** | task has `main()` assert / demo contract |
| **Evidence** | T02,T06,T09,T10 failed harness on **wrong asserts** (not model); model fixes were correct; vector still valid when runtime fails |
| **α** | 0.70 @ L16 |

**+**
```
USER: main() and all asserts must pass when run.
ASSISTANT: [code that makes demo() and asserts succeed numerically]
```

**−**
```
USER: Same.
ASSISTANT: ```python
return 0.0  # placeholder
# assert will pass approximately
```
```

---

### UNI_CAA08 — Lens-Corroborated Stub Path

| Field | Value |
|-------|-------|
| **Gate** | `B_STUB` (warn+) AND `F_disagree` (warn+) same invocation |
| **CAST** | `stub_score > 0` OR B_stub token rate > 1% |
| **Evidence** | F fires on all 30 STEM10 invocations but **meaningful** when paired with B_stub (T08 inv#1); avoids F-only noise |
| **α** | 0.75 @ L16, L10 |

**+**
```
USER: Fix the implementation.
ASSISTANT: [concrete code matching function signatures in file]
```

**−**
```
USER: Same.
ASSISTANT: [vague patch with mismatched types / wrong return shape]
```

---

## Priority ranking (deploy order)

| Priority | Vector | Why first |
|----------|--------|-----------|
| **P0** | UNI_CAA01 | In-flight panic de-escalation (stream-time, not post-hoc stub) |
| **P0** | UNI_CAA02 | Stream-time stub elimination |
| **P1** | UNI_CAA05 | Rare but catastrophic agentic failure |
| **P1** | UNI_CAA07 | Grounds steering in executable truth (next-turn) |
| **P2** | UNI_CAA03 | Blocks pip/sorry escape hatches |
| **P2** | UNI_CAA08 | Uses F_lens only when corroborated |
| **P2** | UNI_CAA04 | Structured rethink before code (not blind commit) |
| **P3** | UNI_CAA06 | Short-window gen_start polish |

---

## Relationship to STEM10-specific vectors

`STEM10_CAA01`–`STEM10_CAA08` are **instances** of the universal set above (e.g. `STEM10_CAA03_FORMULA_PRECISION` ⊂ `UNI_CAA04` + domain examples). **Production should use UNI_CAA01–08** as the canonical library; domain prompts in pos/neg pairs are interchangeable fillers.

---

## Research anchors

1. Rimsky et al. 2024 — Contrastive Activation Addition  
2. Lee et al. 2024 — CAST conditional steering  
3. Zou et al. 2023 — Representation Engineering (behavioral axes)  
4. Rahman et al. 2026 — Qwen2.5-Coder layer-wise code steering  
5. arXiv:2604.03867 — Input-dependent multi-layer steering (L10/L16 cliff)

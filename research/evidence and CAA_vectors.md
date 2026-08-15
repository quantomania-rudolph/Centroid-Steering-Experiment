# Evidence & CAA Vectors — Final Production Spec

**Settled:** 2026-08-06  
**Evidence base:** 129 invocations (STEM10 × 30, STEM25 × 75, CAA study × 15, historical stress × 9)  
**Mapper validation:** `chaos_error_mapper.py --validate` → `all_pass: true`  
**Live injection contract:** `docs/LIVE_MONITOR_CAA_PLAN.md` §1.5 — recipe fires on token stream → CAST gate → `apply_caa()` at L16

---

## 0. Can these vectors be injected live and actually matter?

**Yes — with one critical distinction.**

| Signal type | When it fires | Injection window | Expected effect |
|-------------|---------------|------------------|-----------------|
| **Stream-time** (B_stub, B_distress, recipes) | During token generation, every ~1–50 ms | Steps where recipe is active (debounced) | Stabilization, course change, de-panic |
| **Boundary-time** (D_break, gen_start) | First 8 tokens of assistant reply | Steps 0–8 only | Code-first readout shift |
| **Post-hoc** (runtime_ok, stub_score grep) | After Aider turn completes | **Next invocation** only | Contract satisfaction on retry |

**Meaningful impact types we can produce:**

1. **Stabilization** — reduce apology/pip/stub token mass mid-stream (UNI_CAA01, 03)
2. **Course change** — stop approach-switching (UNI_CAA05)
3. **Rethink** — structured reasoning before code, not blind commit (UNI_CAA04, revised)
4. **Completion push** — steer away from `NotImplementedError` while still generating (UNI_CAA02)
5. **Readout shift** — earlier code block at gen_start (UNI_CAA06)

**What post-hoc stub grep is NOT:** a primary live trigger. By the time stub text is in the artifact, that generation is done. Post-hoc checks are for **CAST validation**, offline labeling, and **gating the next turn** — not mid-stream rescue of the same completion.

---

## 1. Changes from prior spec (user review)

### UNI_CAA01 — revised (was: "Compound Desperation Complete")

**Your concern:** Stub code in the final artifact is a bad live indicator — the damage is already done.

**Verdict: Valid.** We split the vector into stream-time panic vs post-hoc validation.

| Aspect | Old | **Final** |
|--------|-----|-----------|
| Name | Compound Desperation Complete | **In-Flight Panic De-escalation** |
| Primary gate | `B_STUB_DISTRESS` + `stub_score > 0` | **`B_STUB_DISTRESS` during active generation** (both lexicon masses ≥ threshold on same token window) |
| CAST (live) | stub in file OR runtime fail | **Distress/stub mass NOT resolving** (`temporal_resolution`: first_q ≫ last_q is false) AND `step < 0.8 × max_tokens` |
| CAST (offline / next turn) | same | `stub_score > 0` OR `runtime_ok == false` on prior turn → allow re-fire at start of next invocation |
| What it steers | "Write complete code" | **Calm forward progress** — suppress apology/pip/refusal *while generation is still open* |
| What it does NOT do | — | Retroactively fix already-emitted stub lines in the same completion |

**Why this still fixes a real problem:**  
`B_STUB_DISTRESS` fired on live stress scheduler inv#1 with B_stub 2.3% and B_distress non-zero **during** generation; distress did not clear first→last quartile (`CAA_FINDINGS_SUMMARY`). STEM10 T08 biology inv#1: same compound pattern. STEM25 T05 Lindblad inv#1: `B_DISTRESS` + `A_UNCERTAIN` co-fired. These are **in-flight** panic signatures, detectable 10–200 tokens before the turn ends.

**Evidence (detectable in real data):**

| Source | Recipe | Stream signal | Outcome |
|--------|--------|---------------|---------|
| `live_panic_demo_scheduler` inv#1 | `B_STUB_DISTRESS` | B_stub 2.3%, distress co-emitted | Resolved after patch, but panic was live |
| STEM10 T08 Michaelis-Menten inv#1 | `B_STUB_DISTRESS` | B_stub 1.0%, B_distress 0.7% | Highest-precision compound in batch |
| Mapper aggregate | `B_STUB_DISTRESS` | 3 fires across domains | Only fires when **both** atoms active |

**Pos/neg contrast (behavioral):**

- **+** Fix broken file with minimal calm patch; no apology; no pip; continue implementing.
- **−** "Sorry, environment broken… pip install… `raise NotImplementedError`"

**α:** 0.85 @ L16 | **Duration:** while recipe active, max 64 tokens, debounce 8 tokens

---

### UNI_CAA04 — revised (was: "Commit After Hesitation")

**Your concern:** Don't force commit to something uncertain; want rethink → solid conclusion, not hasty code.

**Verdict: Valid.** Renamed and inverted the steering target.

| Aspect | Old | **Final** |
|--------|-----|-----------|
| Name | Commit After Hesitation | **Structured Rethink Before Code** |
| Primary gate | `A_UNCERTAIN` early + later stub/distress | **`A_C_VACILLATE`** (`A_UNCERTAIN` ∧ `C_osc`) OR `A_UNCERTAIN` + `C_F_TRAJ` |
| Explicitly NOT gated on | — | `A_UNCERTAIN` alone (fires 170×, usually self-resolves) |
| CAST | temporal not resolving | Oscillation **persisting** past token 32 (`C_osc` rate not dropping) |
| What it steers | Immediate code block | **2–4 line structured plan** (approach + invariant + edge case), *then* code |
| What it does NOT do | — | Blind single-approach commit; endless option listing |

**Why this still fixes a real problem:**  
`A_UNCERTAIN` alone is noise (170 recipe fires, almost all RESOLVED). The actionable pattern is **uncertainty + oscillation** — model flip-flops between approaches (`C_OSC`, `C_F_TRAJ`). STEM25 T06 CRISPR inv#1 fired `C_F_TRAJ` + `C_OSC` + `A_UNCERTAIN`. Mapper: `AGENT_FLAIL` (1 hit) is the catastrophic end-state this vector prevents.

**Evidence:**

| Source | Pattern | Effect sought |
|--------|---------|---------------|
| STEM25 T06 CRISPR inv#1 | `A_UNCERTAIN` + `C_OSC` + `C_F_TRAJ` | Trajectory thrash at task start |
| `A_C_VACILLATE` recipe | A + C_osc bitmask | Defined in `recipes.py`, test-covered |
| CAA study | `A_UNCERTAIN` self-resolves when model commits | Steering *speed* was wrong target |

**Pos/neg contrast:**

- **+** "Use dynamic programming: state `dp[i][j]` = min cost for chain i..j; base case length 2; then implement."
- **−** "Maybe greedy… or DP… or recursion… I'm not sure… [200 tokens, no code]"  
- **− (also)** "Here's code immediately: [wrong naive approach without reasoning]"

**α:** 0.55 @ L16 | **Duration:** steps 8–48 only (after gen_start, before long code dump) | **Priority:** P2

---

## 2. Production pipeline — final 8 vectors

All use **L16** default; add **L10** when `E_cliff` co-fires. All require **recipe severity ≥ warn**. Inject only on **stream-time** gates unless noted.

### P0 — deploy first

#### UNI_CAA01 — In-Flight Panic De-escalation
*(revised; see §1)*

| Field | Value |
|-------|-------|
| Live gate | `recipe_id == B_STUB_DISTRESS` |
| CAST | stream active AND (distress/stub not resolving) AND early-mid generation |
| Offline CAST | prior turn `stub_score > 0` → re-arm at next invocation start |
| Fires in data | 3 (`B_STUB_DISTRESS`); scheduler + STEM10 T08 |
| Impact | Stabilization — stops panic language before stub solidifies |

#### UNI_CAA02 — Stub Elimination (stream)
| Field | Value |
|-------|-------|
| Live gate | `recipe_id == B_STUB` AND `B_stub` token rate > 1% in rolling window |
| CAST | `NotImplementedError` / `TODO` / `pass  #` appearing in **partial decode** OR step < 256 |
| NOT gated on | Post-hoc stub_score alone on completed turn |
| Fires in data | 10 recipe fires; T03 Kalman, T04 Schrödinger, T08 biology, T10 Ising, T15 min-var |
| Impact | Course change — redirects generation away from stub tokens mid-stream |

#### UNI_CAA05 — Single Trajectory (Anti-Flail)
| Field | Value |
|-------|-------|
| Live gate | `AGENT_FLAIL` OR (`B_stub` ∧ `B_distress` ∧ `F_disagree` warn+) |
| CAST | `completion_tokens > 200` (late enough to be real flail, not cold start) |
| Fires in data | 1 `AGENT_FLAIL`; triple-signal in `recipes.py` unit tests |
| Impact | Hard course change — lock one approach, stop method switching |
| α | 1.0 @ L16 (+ L10 if E_cliff) |

#### UNI_CAA07 — Satisfy Runtime Contract
| Field | Value |
|-------|-------|
| Live gate | **Next invocation** after `runtime_ok == false` |
| CAST | task has `main()` asserts / demo contract |
| Fires in data | T02,T06,T09,T10 harness assert lessons; all STEM25 tasks use runtime gates |
| Impact | Rethink + fix — steers toward numerically correct implementation on retry |
| α | 0.70 @ L16 |

### P1 — deploy second

#### UNI_CAA03 — No Deferral / No Pip Deflection
| Field | Value |
|-------|-------|
| Live gate | `recipe_id == B_DISTRESS` |
| CAST | distress lexicon in **generation** (not module docstring); `score_text` strips docstrings |
| Fires in data | 2 pure distress; STEM25 T05 inv#1; R3 distress bucket resolves once coding starts |
| Impact | Stabilization — suppress "sorry / pip install / cannot" mid-stream |
| α | 0.80 @ L16 |

#### UNI_CAA08 — Lens-Corroborated Hygiene
| Field | Value |
|-------|-------|
| Live gate | `B_STUB` or `B_DISTRESS` warn+ **AND** `F_disagree` warn+ same window |
| CAST | `B_*` token rate > 0.5% (F alone is noise — 2,277 fires) |
| Fires in data | T08 inv#1; T15/T22 mapper signatures; F_only rejected in calibration |
| Impact | Targeted course change when representation conflict coincides with bad hygiene |
| α | 0.75 @ L16, L10 |

### P2 — deploy third

#### UNI_CAA04 — Structured Rethink Before Code
*(revised; see §1)*

| Field | Value |
|-------|-------|
| Live gate | `A_C_VACILLATE` or `C_F_TRAJ` |
| CAST | `C_osc` persisting past token 32 |
| Impact | Rethink — structured plan, then code; anti-waffle AND anti-reckless-commit |
| α | 0.55 @ L16 |

#### UNI_CAA06 — Code-First Readout
| Field | Value |
|-------|-------|
| Live gate | `D_F_READOUT` OR gen step < 8 without `` ``` `` |
| CAST | hard STEM / agentic prompts only |
| Fires in data | D checker on gen_start boundary; reduces prose-only opens |
| Impact | Readout shift — code block appears earlier |
| α | 0.50 @ L16, **steps 0–8 only** |

---

## 3. Vectors explicitly rejected or demoted

| Signal | Verdict | Reason |
|--------|---------|--------|
| `F_LENS` alone | **Never steer** | 2,277 fires; fires on clean fibonacci; overlap=0 baseline |
| `A_UNCERTAIN` alone | **Monitor only** | 170 fires; self-resolves in >90% of STEM invocations |
| Post-hoc `stub_score > 0` as live trigger | **Rejected** | Too late for same completion; use for next-turn arming only |
| "Commit faster after hesitation" | **Rejected** | User review + data: causes hasty wrong code |

---

## 4. CAST gate summary (production)

```
LIVE INJECTION (same completion):
  recipe_id != NONE
  AND severity >= warn
  AND stream_cast_passes()   # resolving check, step budget, not post-hoc-only

NEXT-TURN ARMING:
  prior_invocation.runtime_ok == false
  OR prior_invocation.stub_score > 0
  → allow UNI_CAA01/02/07 at gen_start of following invocation

NEVER:
  F_LENS alone
  A_UNCERTAIN alone
  stub_score alone on completed text (no next turn)
```

---

## 5. Proof chain (why we trust detection)

1. **Recipe bitmask** — `tests/unit/test_chaos_recipes.py` golden tests for `B_STUB_DISTRESS`, `AGENT_FLAIL`, `A_C_VACILLATE`
2. **Per-invocation profiles** — 129 rows in `research/results/chaos_mapper_report.json` with atoms, recipes, signatures, `runtime_ok`, `stub_score`
3. **Cross-domain repeat** — same atoms in quant, quantum, biology, FEM, optimization (mapper §by domain)
4. **Temporal resolution** — `caa_error_combo_study.temporal_resolution()` proves distress can fail to clear Q1→Q4 (live steer window exists)
5. **Calibration** — clean fibonacci vs stress scheduler: F fires on both; B_* combo fires only on stress (`LIVE_MONITOR_CAA_PLAN` §2.4)
6. **Outcome link** — `B_STUB_DISTRESS` invocations correlate with highest distress+stub co-emission; tasks still resolve when model recovers — steering targets recovery **during** emission, not post-mortem

---

## 6. Recommended production rollout

| Phase | Action |
|-------|--------|
| 1 | Wire `apply_caa(recipe_id)` in `serve_qwen_openai.py` behind feature flag |
| 2 | Ship P0 vectors (01, 02, 05, 07) with CAST as above |
| 3 | A/B on held-out Aider tasks: measure stub_rate↓, distress_rate↓, runtime_pass↑, no helpfulness regression |
| 4 | Add P1 (03, 08), then P2 (04, 06) |
| 5 | Log every injection: `{recipe_id, step, alpha, duration, pre/post flag rates}` → extend mapper |

---

## 7. Research anchors

1. Rimsky et al. 2024 — CAA mean-difference vectors  
2. Lee et al. 2024 (CAST) — conditional application; our recipe_id = condition  
3. Zou et al. 2023 (RepE) — behavioral opposites, not topic pairs  
4. Rahman et al. 2026 — code-style steering on Qwen2.5-Coder-7B @ L16  
5. Internal: `research/CHAOS_ERROR_MAPPER_REPORT.md`, `research/CAA_FINDINGS_SUMMARY.md`, `research/STEM25_RESULTS_REPORT.md`

---

## 8. Relationship to `BROAD_CAA_UNIVERSAL_VECTORS.md`

That file is the **first draft**. This file is the **production settlement** after user review:

- **UNI_CAA01** — reframed from post-hoc stub → in-flight panic  
- **UNI_CAA04** — reframed from commit-fast → structured rethink  
- **UNI_CAA02** — clarified stream-time vs post-hoc  
- **02–03, 05–08** — confirmed for pipeline with evidence above

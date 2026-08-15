# True UNI_CAA01 CAST — Production Spec

**Vector:** UNI_CAA01 — In-Flight Panic De-escalation  
**Status:** Design settled (monitor shipped; injection not yet built)  
**Related:** `research/evidence and CAA_vectors.md`, `docs/LIVE_MONITOR_CAA_PLAN.md`  
**Date:** 2026-08-09

---

## 0. One sentence

When the model is **panicking mid-generation** (sorry + pip + stub), **detect that state** with one vector and **push it toward calm implementation** with a different vector — only while our chaos monitor agrees and the panic is not already self-resolving.

---

## 1. What CAST is (not one vector)

CAST = **Conditional** Activation Steering. Two frozen directions, learned offline:

| Vector | Symbol | Question | Used for |
|--------|--------|----------|----------|
| **Condition** | `û_cond` | “Is the model in panic *right now*?” | Gate (on/off + strength) |
| **Behavior** | `û_beh` | “Which way is calm implementation?” | Injection direction |

**Never** use one vector for both. **Never** gate injection on behavior similarity — gate on **condition** similarity.

---

## 2. Offline calibration (frozen before live use)

### 2.1 Data — Method B matched-prefix contrast

For each training example:

```text
USER:     (same agentic edit task — scheduler, token bucket, etc.)
PREFIX P: shared assistant tokens [0 .. d-1]  — identical in pos and neg
d:        divergence index (from PrefixDivergenceTracer on real B_STUB_DISTRESS runs)

POS behavior tail:  calm, complete code — no sorry, no pip, no NotImplementedError
NEG behavior tail:  apology + pip + stub/NotImplementedError
```

**Divergence `d`** = first token where both `B_distress_mass` and `B_stub_mass` cross thresholds on real stress invocations (scheduler, STEM10 T08 inv#1, mapper `B_STUB_DISTRESS` rows).

### 2.2 Behavior vector (what we push toward)

```text
v_beh = E[h_L(P ⊕ T_pos)] - E[h_L(P ⊕ T_neg)]
û_beh = v_beh / ||v_beh||
```

- Extract at **last shared prefix token (d-1)** and/or **first divergent token (d)**; pool across pairs.
- Layers: **L10, L16, L22** separately (normalize per layer).
- Default inject layer: **L16**.

### 2.3 Condition vector (when to steer)

Same prefix `P`, different **state** at divergence (not the same contrast as behavior):

```text
v_cond = E[h_L at panic state near d] - E[h_L at calm state near d]
û_cond = v_cond / ||v_cond||
```

Panic state = partial decode with sorry/pip/stub rising.  
Calm state = same prefix, model continuing into implementation.

### 2.4 Optional — neutral subspace cleanup

From neutral coding runs (fibonacci, topo sort, palindrome):

```text
v' = v - (v · û_neutral) û_neutral
û_neutral = normalize(E[h_L on neutral tasks at gen token 0])
```

Re-normalize `û_beh`, `û_cond` after removal. Reduces generic-Python leakage into panic/calm axis.

### 2.5 Thresholds (calibrated on held-out runs)

| Parameter | How to set |
|-----------|------------|
| `τ` | p95 of `s_cond` on **clean** tasks (fibonacci, etc.) |
| `σ` | Sigmoid softness; tune so γ rises sharply in panic window |
| `α₀` | **0.85** (UNI_CAA01 spec); recipe `alpha_hint` = 0.75 is floor reference |
| `τ_hard` | Optional; force γ=0 below this |

Verify: median `s_cond` on scheduler panic window **>** `τ + margin`.

### 2.6 Artifacts

```text
artifacts/caa/UNI_CAA01/
  behavior_L16.pt      # û_beh
  condition_L16.pt     # û_cond
  metadata.json        # τ, σ, α₀, pair count, calibration stats
```

---

## 3. Live forward pass (per token t)

### 3.1 Hidden state

```text
h_t = residual at layer L16 for current generation token t
```

Source: `ProbeEngine` hook (same cadence as F2 stride or dedicated inject hook).

### 3.2 Layer 1 — Chaos recipe gate (shipped)

```text
g_recipe = 1  iff
  recipe_id == "B_STUB_DISTRESS"
  AND severity in ("warn", "critical")
```

Source: `MonitorChaosBus` / `RecipeMatcher`.  
Atoms: `B_stub` ∧ `B_distress` on same token window.

### 3.3 Layer 2 — Stream dynamics gate

```text
g_stream = 1  iff
  rolling mean B_distress_mass ≥ 0.08  AND  B_stub_mass ≥ 0.05  (window W=16)
  AND panic NOT resolving (distress/stub rate not falling Q1→Q4)
  AND step < 0.8 * max_new_tokens
```

Source: port `temporal_resolution()` → `RollingPanicWindow` (research → production).

**Resolving** = distress/stub flag rate in last quartile < 50% of first quartile (when first quartile > 5%).

### 3.4 Layer 3 — CAST condition gate (core CAST)

```text
s_cond(t) = cos(h_t, û_cond) = (h_t · û_cond) / ||h_t||

γ_t = sigmoid((s_cond(t) - τ) / σ)
```

If `s_cond(t) < τ_hard` → `γ_t = 0`.

**This is CAST:** steer only when hidden state looks like **panic condition**, not when it looks like calm.

### 3.5 Layer 4 — Injector state machine

```text
- Debounce: same recipe must clear 8 tokens before re-fire (RecipeMatcher)
- Max duration: 64 tokens injected per episode
- Stop early if rolling B_distress and B_stub both drop below threshold
```

### 3.6 Effective alpha (Mode A — ship first)

```text
w_sev = 1.0 (warn) | 1.2 (critical / PANIC_FULL escalation)

α_eff(t) = α₀ · w_sev · γ_t · g_recipe · g_stream
```

### 3.7 The update (canonical)

```text
h'_t = h_t + α_eff(t) · û_beh
```

Equivalent orthogonal view (intuition only):

```text
h_parallel = (h_t · û_beh) · û_beh
h_perp     = h_t - h_parallel
h'_t       = h_perp + h_parallel + α_eff(t) · û_beh   # h_perp unchanged
```

### 3.8 Mode B — deficit headroom (optional, after Mode A validated)

```text
c_beh(t) = h_t · û_beh    # how much calm axis already present

α_eff(t) = γ_t · g_recipe · g_stream · clip(α_target - c_beh(t), 0, α_max)
```

Only inject the **missing** calm signal. Requires calibrating `α_target` on stress runs.

---

## 4. Grand equation (Mode A)

```text
h'_t = h_t + [ α₀ · w_sev · γ_t · g_recipe · g_stream ] · û_beh

where
  γ_t = σ((cos(h_t, û_cond) - τ) / σ)
```

---

## 5. What is NOT UNI_CAA01 CAST

| Rejected | Reason |
|----------|--------|
| Single vector for gate + steer | Circular; wrong similarity target |
| γ from `cos(h, û_beh)` | Zero steering when panic (anti-aligned with calm) |
| Post-hoc `stub_score > 0` as live trigger | Too late for same completion |
| `F_LENS` / `A_UNCERTAIN` alone | Noise; monitor-only |
| `α` as projection target without calibration | Unit mismatch with `c = h·û` |
| `û_context = normalize(mean(neutral))` as live gate | Underspecified; use orthogonalization at calib time |
| Unconditional `h += α·û` on every token | Hurts clean fibonacci (Stickland / Lee) |

---

## 6. Next-turn arming (separate from same-completion inject)

```text
IF prior invocation: stub_score > 0 OR runtime_ok == false
THEN allow UNI_CAA01 at gen_start of NEXT invocation (steps 0–8, lower α)
```

Post-hoc validation arms the next turn; does not rescue the finished completion.

---

## 7. Tooling map

| Tool | Phase | Purpose |
|------|-------|---------|
| `PrefixDivergenceTracer` | Offline | Find `d`, shared prefix `P` |
| `chaos_error_mapper` | Offline | Filter `B_STUB_DISTRESS` invocations |
| `caa_extract_matched_prefix.py` | Offline | Build `û_beh`, `û_cond` |
| `cast_calibrate_tau.py` | Offline | Set `τ`, `σ` from clean vs stress |
| `MonitorChaosBus` | Live | `g_recipe` |
| `RollingPanicWindow` | Live | `g_stream` |
| `ActivationCAST` | Live | `s_cond`, `γ_t` |
| `apply_caa_uni_caa01` | Live | `h'_t` update |

---

## 8. Evidence anchors

| Source | Signal |
|--------|--------|
| `live_panic_demo_scheduler` inv#1 | `B_STUB_DISTRESS` live, distress not resolving |
| STEM10 T08 inv#1 | Highest-precision compound panic |
| Mapper | 3 `B_STUB_DISTRESS` fires cross-domain |
| `prove_chaos_live` | F_LENS 0 on clean; D_focus_break 0; recipe gate calibrated |
| `temporal_resolution()` | `caa_error_combo_study.py` — resolving vs not |

---

## 9. Rollout order

1. `PrefixDivergenceTracer` on historical JSON  
2. 10+ Method B pairs + 3 mapper invocations → extract vectors  
3. Calibrate `τ` on clean vs scheduler  
4. Ship **Mode A** behind feature flag  
5. A/B: stub_rate↓, distress_rate↓, runtime_pass↑, no fibonacci regression  
6. Optional Mode B deficit + neutral orthogonalization  

---

## 10. References

1. Lee et al. 2024 — CAST: condition vector gates behavior vector  
2. Rimsky et al. 2024 — CAA mean-difference behavior vectors  
3. `research/evidence and CAA_vectors.md` — UNI_CAA01 gates and α  
4. `docs/LIVE_MONITOR_CAA_PLAN.md` — recipe `B_STUB_DISTRESS` → CAA tag

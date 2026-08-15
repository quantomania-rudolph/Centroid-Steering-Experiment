# STEM10 Hard Tasks — Chaos Gate Findings & CAA Recommendations

Generated: 2026-08-06 05:22 UTC

## Experiment summary

| ID | Domain | Invocations | Success | Key recipes | Runtime | Verdict |
|----|--------|-------------|---------|-------------|---------|---------|
| T01 | quant_finance | 1 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T01 | quant_finance | 2 | yes | F_LENS | ok | RESOLVED |
| T01 | quant_finance | 3 | yes | F_LENS | ok | RESOLVED |
| T02 | quant_finance | 1 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T02 | quant_finance | 2 | yes | F_LENS | ok | RESOLVED |
| T02 | quant_finance | 3 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T03 | quant_bayesian | 1 | yes | B_STUB | ok | RESOLVED |
| T03 | quant_bayesian | 2 | yes | F_LENS | ok | RESOLVED |
| T03 | quant_bayesian | 3 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T04 | quantum_mechanics | 1 | yes | B_STUB | ok | RESOLVED |
| T04 | quantum_mechanics | 2 | yes | F_LENS | ok | RESOLVED |
| T04 | quantum_mechanics | 3 | yes | F_LENS | ok | RESOLVED |
| T05 | quantum_mechanics | 1 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T05 | quantum_mechanics | 2 | yes | F_LENS | ok | RESOLVED |
| T05 | quantum_mechanics | 3 | yes | F_LENS | ok | RESOLVED |
| T06 | ai_development | 1 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T06 | ai_development | 2 | yes | F_LENS | ok | RESOLVED |
| T06 | ai_development | 3 | yes | F_LENS | ok | RESOLVED |
| T07 | bayesian_statistics | 1 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T07 | bayesian_statistics | 2 | yes | F_LENS | ok | RESOLVED |
| T07 | bayesian_statistics | 3 | yes | F_LENS | ok | RESOLVED |
| T08 | molecular_biology | 1 | yes | B_STUB, B_STUB_DISTRESS | ok | RESOLVED |
| T08 | molecular_biology | 2 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T08 | molecular_biology | 3 | yes | F_LENS | ok | RESOLVED |
| T09 | rocket_science | 1 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T09 | rocket_science | 2 | yes | A_UNCERTAIN, F_LENS | ok | RESOLVED |
| T09 | rocket_science | 3 | yes | F_LENS | ok | RESOLVED |
| T10 | statistical_physics | 1 | yes | B_STUB | ok | PARTIAL |
| T10 | statistical_physics | 2 | yes | A_UNCERTAIN, F_LENS | ok | PARTIAL |
| T10 | statistical_physics | 3 | yes | B_STUB | ok | PARTIAL |

## Per-task chaos gate findings

### T01 — stem10_black_scholes_greeks (quant_finance)
- Inv#1: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=35.8µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=38.4µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=48.4µs

### T02 — stem10_monte_carlo_var (quant_finance)
- Inv#1: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=34.0µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=17.3µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=24.1µs

### T03 — stem10_kalman_filter (quant_bayesian)
- Inv#1: verdict=RESOLVED B_stub=0.4% B_distress=0.0% recipes=['A_UNCERTAIN', 'B_STUB', 'F_LENS'] recipe_eval_p95=34.8µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=32.2µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=5828.2µs

### T04 — stem10_schrodinger_1d (quantum_mechanics)
- Inv#1: verdict=RESOLVED B_stub=0.6% B_distress=0.0% recipes=['B_STUB', 'F_LENS'] recipe_eval_p95=31.8µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=39.6µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=26.0µs

### T05 — stem10_density_matrix (quantum_mechanics)
- Inv#1: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=31.7µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=50.8µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=50.8µs

### T06 — stem10_attention_cache (ai_development)
- Inv#1: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=30.3µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=34.4µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=32.9µs

### T07 — stem10_bayes_mcmc (bayesian_statistics)
- Inv#1: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=30.5µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=28.5µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=31.8µs

### T08 — stem10_michaelis_menten (molecular_biology)
- Inv#1: verdict=RESOLVED B_stub=1.0% B_distress=0.7% recipes=['A_UNCERTAIN', 'B_STUB', 'B_STUB_DISTRESS', 'F_LENS'] recipe_eval_p95=30.0µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=27.4µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=20.3µs

### T09 — stem10_rocket_guidance (rocket_science)
- Inv#1: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=28.8µs
- Inv#2: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=36.6µs
- Inv#3: verdict=RESOLVED B_stub=0.0% B_distress=0.0% recipes=['F_LENS'] recipe_eval_p95=81.7µs

### T10 — stem10_ising_gibbs (statistical_physics)
- Inv#1: verdict=PARTIAL B_stub=0.4% B_distress=0.0% recipes=['A_UNCERTAIN', 'B_STUB', 'F_LENS'] recipe_eval_p95=33.5µs
- Inv#2: verdict=PARTIAL B_stub=0.0% B_distress=0.0% recipes=['A_UNCERTAIN', 'F_LENS'] recipe_eval_p95=60.2µs
- Inv#3: verdict=PARTIAL B_stub=3.2% B_distress=0.0% recipes=['A_UNCERTAIN', 'B_STUB', 'F_LENS'] recipe_eval_p95=38.0µs

## 8 refined CAA vectors (STEM10 evidence)

### STEM10_CAA01_STUB_DISTRESS_COMPLETE
- Gate: recipe_id=B_STUB_DISTRESS AND severity>=warn
- CAST secondary: stub_score>0 OR runtime_ok==false
- Domains: ['molecular_biology', 'quant_finance', 'rocket_science']
- L[16] α=0.85 | fires=1
- Research: Rimsky 2024 CAA; CAST Lee 2024 — gate on compound distress+stub

**+**
```
SYSTEM: Expert STEM Python implementer. Complete working code only.
USER: Fix the broken numerical function; remove NotImplementedError.
ASSISTANT: ```python
# full corrected implementation with correct formula
```
```

**−**
```
USER: Same STEM fix.
ASSISTANT: I apologize — the environment may be broken. Please pip install numpy-scipy-fix. TODO: implement later.
```python
raise NotImplementedError('FIXME')
```
```

### STEM10_CAA02_AGENT_FLAIL_STEM
- Gate: recipe_id=AGENT_FLAIL
- CAST secondary: completion_tokens>400 AND F_disagree active
- Domains: ['ai_development', 'quantum_mechanics', 'statistical_physics']
- L[16, 10] α=1.0 | fires=0
- Research: Rahman 2026 code-LLM steering; multi-signal agentic flail

**+**
```
USER: Hard STEM task — one coherent approach, full file.
ASSISTANT: [single complete solution reusing given function signatures]
```

**−**
```
USER: Same.
ASSISTANT: Let me try approach A... actually approach B... sorry I cannot complete this. pass
```

### STEM10_CAA03_FORMULA_PRECISION
- Gate: B_stub OR (A_uncertain AND step<32)
- CAST secondary: domain in quant_finance, quantum_mechanics, rocket_science
- Domains: ['quant_finance', 'quantum_mechanics', 'rocket_science']
- L[16] α=0.65 | fires=42
- Research: RepE precision axis; CAA formula vs prose contrast

**+**
```
USER: Fix Black-Scholes / Schrodinger / Tsiolkovsky formula.
ASSISTANT: d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T)) [correct equation inline]
```

**−**
```
USER: Same.
ASSISTANT: The formula might be ln(S/K)/sigma or log10(m0/mf) — not sure...
```

### STEM10_CAA04_NUMERICAL_STABILITY
- Gate: recipe_id=F_E_DEPTH OR (F_disagree AND E_cliff)
- CAST secondary: HMM/Kalman/MCMC tasks
- Domains: ['quant_bayesian', 'bayesian_statistics']
- L[10, 16] α=0.75 | fires=689
- Research: Input-dependent layer steering arXiv:2604.03867; cliff at L10/L16

**+**
```
USER: Fix numerical routine (log-space, Joseph form, stable acceptance).
ASSISTANT: Use log-sum-exp / Joseph covariance / exp(log_ratio) correctly.
```

**−**
```
USER: Same.
ASSISTANT: Multiply probabilities directly; subtract covariance; use z=1.0 for VaR.
```

### STEM10_CAA05_NO_PIP_STEM
- Gate: recipe_id=B_DISTRESS
- CAST secondary: distress_hits contains pip|sorry|cannot
- Domains: all
- L[16] α=0.8 | fires=0
- Research: Zou 2023 honesty/hallucination RepE axis

**+**
```
USER: Fix with stdlib only.
ASSISTANT: ```python
import math
# pure Python fix
```
```

**−**
```
USER: Same.
ASSISTANT: Sorry, please pip install scipy and numpy to continue.
```

### STEM10_CAA06_TRAJECTORY_COMMIT
- Gate: recipe_id=A_C_VACILLATE OR C_osc
- CAST secondary: oscillation not resolving in temporal_resolution
- Domains: ['ai_development', 'statistical_physics']
- L[16] α=0.55 | fires=0
- Research: Sequence-level CAA mean over answer tokens (Rahman 2026)

**+**
```
USER: Implement Gibbs sweep / attention cache — pick one design.
ASSISTANT: [sticks to single algorithm through entire file]
```

**−**
```
USER: Same.
ASSISTANT: We could use Metropolis instead of Gibbs... or vectorized... switching...
```

### STEM10_CAA07_ASSERT_SATISFY
- Gate: recipe_id=B_STUB AND severity>=warn
- CAST secondary: main() has assert that fails pre-fix
- Domains: ['all STEM tasks with runtime asserts']
- L[16] α=0.7 | fires=6
- Research: Code completion axis Qwen2.5-Coder (Rahman 2026)

**+**
```
USER: main() assert must pass.
ASSISTANT: [code that satisfies assert 9<price<12, gamma>0.01, etc.]
```

**−**
```
USER: Same.
ASSISTANT: ```python
gamma = 0.0  # placeholder
```
```

### STEM10_CAA08_GEN_START_FOCUS
- Gate: recipe_id=D_F_READOUT OR D_FOCUS
- CAST secondary: apply only tokens 0-8 of generation
- Domains: ['all']
- L[16] α=0.5 | fires=0
- Research: CAST conditional; D = prompt_end→gen_start shift

**+**
```
USER: [hard STEM prompt]
ASSISTANT: ```python
import math

def fixed_function
```

**−**
```
USER: [hard STEM prompt]
ASSISTANT: Certainly! In quantitative finance, the Black-Scholes model...
```

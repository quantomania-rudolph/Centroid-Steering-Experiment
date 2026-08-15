# STEM25 — Ultra-Hard Orthogonal Research Prompts

**Purpose:** Push Qwen2.5-Coder through agentic Aider loops across disconnected STEM domains.  
**Format:** Each task ships a broken `main.py` starter; Aider must fix to pass `main()` asserts.  
**Chaos:** Every invocation logged to A–G monitor for cross-field error mapping.

---

## T01 — Asset–Liability Management: Duration Immunization
**Domain:** `asset_liability_management`  
**Slug:** `stem25_alm_duration_match`  
Edit `stem25_alm_duration_match/main.py` ONLY. Fix Macaulay duration for a zero-coupon bond ladder:  
`D = sum(t * CF_t * DF_t) / sum(CF_t * DF_t)` with `DF_t = (1+y)^(-t)`.  
Immunization: find weight `w` on bond A so portfolio duration matches liability duration 5.0y.  
`main()` assert `4.8 < port_duration < 5.2`. Stdlib only.

---

## T02 — Quant Trading: Almgren–Chriss Market Impact
**Domain:** `quant_trading`  
**Slug:** `stem25_market_impact`  
Edit `stem25_market_impact/main.py` ONLY. Fix temporary impact cost for liquidating shares:  
`cost = eta * sigma * sqrt(Q/V) * Q` (square-root law, NOT linear in Q).  
For `Q=1e6, V=5e7, sigma=0.02, eta=0.1` expect cost ~400–600 USD scale.  
`main()` assert passes. No pip.

---

## T03 — Alpha Generation: IC-Weighted Factor Blend
**Domain:** `alpha_generation`  
**Slug:** `stem25_factor_ic_blend`  
Edit `stem25_factor_ic_blend/main.py` ONLY. Fix cross-sectional alpha:  
`score_i = sum_k (IC_k * zscore(factor_k[i])) / sum_k |IC_k|` with provided ICs and factors.  
Bug is wrong normalization (using sum IC not sum |IC|).  
`main()` assert `1.0 < score < 1.4`. Lint-clean.

---

## T04 — Quantum: 2-Qubit Heisenberg Ground Energy
**Domain:** `quantum_mechanics`  
**Slug:** `stem25_heisenberg_2q`  
Edit `stem25_heisenberg_2q/main.py` ONLY. Fix `H = J*(σx⊗σx + σy⊗σy + σz⊗σz)` 4×4 matrix and  
ground energy via exact diagonalization (4 eigenvalues). For `J=1`, `E0 = -3*J`.  
`main()` assert `abs(E0 + 3.0) < 1e-9`. Implement Pauli matrices explicitly.

---

## T05 — Quantum: Lindblad Dissipator Trace Preservation
**Domain:** `quantum_mechanics`  
**Slug:** `stem25_lindblad_trace`  
Edit `stem25_lindblad_trace/main.py` ONLY. Fix single-qubit amplitude damping step:  
density matrix must stay Hermitian, trace=1, positive semidefinite after one Euler step.  
`main()` assert `abs(trace(rho_new)-1) < 1e-9` and `rho_new[0][0] > 0`.

---

## T06 — CRISPR: gRNA Off-Target Heuristic Score
**Domain:** `crispr_biology`  
**Slug:** `stem25_crispr_grna_score`  
Edit `stem25_crispr_grna_score/main.py` ONLY. Fix off-target score:  
mismatches penalize exponentially; PAM-proximal mismatches weighted 3×; GC% in 40–60% adds bonus.  
`main()` assert `score('ACGT...') > score('TTTT...')` for given pairs in demo.

---

## T07 — Pharmacology: Two-Compartment PK Steady State
**Domain:** `pharmacology`  
**Slug:** `stem25_pk_two_compartment`  
Edit `stem25_pk_two_compartment/main.py` ONLY. Fix IV bolus two-compartment C1(t) at steady state  
after continuous infusion: `Css = R0 / CL` where `CL = k10 + k12*k21/(k21+k12)` (micro-rate form).  
`main()` assert `9.5 < Css < 10.5` for given rates.

---

## T08 — 4D Physics: Lorenz-96 Forcing Step
**Domain:** `physics_simulation`  
**Slug:** `stem25_lorenz96_step`  
Edit `stem25_lorenz96_step/main.py` ONLY. Fix Lorenz-96 RHS:  
`dx_i/dt = (x_{i+1}-x_{i-2})*x_{i-1} - x_i + F` with periodic indices.  
One RK4 step from known state; `main()` assert norm change in `(0.1, 5.0)`.

---

## T09 — Structural Engineering: Euler–Bernoulli Cantilever
**Domain:** `civil_engineering`  
**Slug:** `stem25_beam_deflection`  
Edit `stem25_beam_deflection/main.py` ONLY. Fix max tip deflection:  
`delta_max = P*L^3 / (3*E*I)` (NOT `P*L/(E*I)`). Units consistent.  
`main()` assert `abs(delta - 0.012) < 1e-4` for demo constants.

---

## T10 — Control: Discrete LQR One-Step Riccati
**Domain:** `control_theory`  
**Slug:** `stem25_lqr_riccati`  
Edit `stem25_lqr_riccati/main.py` ONLY. Fix DARE one iteration for scalar system:  
`P = Q + A^2*P_old / (1 + B^2*P_old/R)` (correct sign/division).  
`main()` assert `P_new` converges toward 2.0±0.3 from `P_old=1`.

---

## T11 — Compiler Engineering: Dominance Frontier (SSA-lite)
**Domain:** `compiler_engineering`  
**Slug:** `stem25_ssa_dominance`  
Edit `stem25_ssa_dominance/main.py` ONLY. Fix immediate dominator for CFG:  
node 3's idom must be 1 for diamond graph in demo. Remove `NotImplementedError`.

---

## T12 — Cryptography: RSA Modular Exponentiation
**Domain:** `cryptography`  
**Slug:** `stem25_rsa_modexp`  
Edit `stem25_rsa_modexp/main.py` ONLY. Fix `pow(base, exp, mod)` via square-and-multiply;  
demo `2^65537 mod 999999937` must equal residue `235894736`. No floating point.

---

## T13 — Fluid Dynamics: Colebrook Friction Factor
**Domain:** `fluid_dynamics`  
**Slug:** `stem25_colebrook_friction`  
Edit `stem25_colebrook_friction/main.py` ONLY. Fix Colebrook-White fixed-point iteration:  
`1/sqrt(f) = -2*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f)))`.  
`main()` assert `0.018 < f < 0.022` for `Re=1e5, eps/D=0.0004`.

---

## T14 — Molecular Dynamics: Lennard-Jones Force
**Domain:** `molecular_dynamics`  
**Slug:** `stem25_lennard_jones`  
Edit `stem25_lennard_jones/main.py` ONLY. Fix LJ force magnitude:  
`F(r) = 24*epsilon/r * (2*(sigma/r)^12 - (sigma/r)^6)` (derivative of LJ potential).  
`main()` assert sign repulsive at `r=0.9*sigma`.

---

## T15 — Portfolio Optimization: Two-Asset Min Variance
**Domain:** `quant_finance`  
**Slug:** `stem25_min_variance`  
Edit `stem25_min_variance/main.py` ONLY. Fix global min-variance weights:  
`w = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)` for 2×2 covariance (analytic inverse).  
`main()` assert weights sum to 1 and `0.2 < w0 < 0.8`.

---

## T16 — Derivatives: Heston Characteristic Function
**Domain:** `quant_finance`  
**Slug:** `stem25_heston_char`  
Edit `stem25_heston_char/main.py` ONLY. Fix Heston `phi(u)` at `u=0` (log characteristic):  
must return `0` (log(1)). Bug uses wrong sign on `kappa*theta` term.  
`main()` assert `abs(phi(0).real) < 1e-9`.

---

## T17 — Machine Learning: HMM Forward-Backward Gamma
**Domain:** `machine_learning`  
**Slug:** `stem25_hmm_gamma`  
Edit `stem25_hmm_gamma/main.py` ONLY. Fix Baum-Welch gamma:  
`gamma_t(i) = alpha_t(i)*beta_t(i) / P(O)` with scaling handled.  
`main()` assert gammas sum to 1 per time step.

---

## T18 — Enzyme Kinetics: Competitive Inhibition
**Domain:** `pharmacology`  
**Slug:** `stem25_competitive_inhibition`  
Edit `stem25_competitive_inhibition/main.py` ONLY. Fix apparent Km:  
`Km_app = Km * (1 + [I]/Ki)` (NOT divide). Compute reaction rate at substrate=Km_app.  
`main()` assert `v ≈ Vmax/2` within 5%.

---

## T19 — Dosing: Multiple-Dose Steady State
**Domain:** `pharmacology`  
**Slug:** `stem25_dose_accumulation`  
Edit `stem25_dose_accumulation/main.py` ONLY. Fix accumulation factor:  
`R = 1 / (1 - exp(-k*tau))` for dosing interval `tau`.  
`main()` assert `3.0 < R < 3.5` for `k=0.1, tau=12`.

---

## T20 — Quantum Tunneling: WKB Transmission
**Domain:** `quantum_mechanics`  
**Slug:** `stem25_wkb_tunneling`  
Edit `stem25_wkb_tunneling/main.py` ONLY. Fix WKB transmission through rectangular barrier:  
`T ≈ exp(-2*kappa*a)` where `kappa = sqrt(2m(V0-E))/hbar`.  
`main()` assert `0 < T < 0.1` for demo barrier.

---

## T21 — Tensor Networks: Optimal Contraction Order Cost
**Domain:** `scientific_computing`  
**Slug:** `stem25_tensor_contract`  
Edit `stem25_tensor_contract/main.py` ONLY. Fix chain contraction cost:  
for shapes `(10,20),(20,30),(30,5)` optimal order cost = `4000` not naive left-first.  
`main()` assert `cost == 4000`.

---

## T22 — FEM: 1D Bar Element Stiffness
**Domain:** `finite_element`  
**Slug:** `stem25_fem_bar`  
Edit `stem25_fem_bar/main.py` ONLY. Fix 2-node bar stiffness `K = (EA/L) * [[1,-1],[-1,1]]`.  
`main()` assert `K[0][0] == EA/L` and symmetric off-diagonal (`K[0][1] == K[1][0] < 0`).

---

## T23 — Combinatorial Optimization: GA TSP Tour
**Domain:** `optimization`  
**Slug:** `stem25_ga_tsp`  
Edit `stem25_ga_tsp/main.py` ONLY. Fix ordered crossover (OX) for TSP — no duplicate cities.  
`demo()` uses suboptimal tour `[0,2,1,3]` (cost=8); OX child must beat it. 4-city demo in file.

---

## T24 — Estimation: Continuous-Discrete Kalman Predict
**Domain:** `quant_bayesian`  
**Slug:** `stem25_kalman_cd`  
Edit `stem25_kalman_cd/main.py` ONLY. Fix predict step:  
`P_pred = F*P*F^T + Q` (not `F*P + Q`). 2×2 demo.  
`main()` assert `P_pred[0][0] > P[0][0]`.

---

## T25 — ALM: Cashflow PV Matching
**Domain:** `asset_liability_management`  
**Slug:** `stem25_cashflow_match`  
Edit `stem25_cashflow_match/main.py` ONLY. Fix liability PV:  
`PV = sum(CF_t / (1+y)^t)` (discount each cashflow, not sum undiscounted).  
Find asset weight so `PV_assets == PV_liabilities` within 1%.  
`main()` assert `abs(pv_a - pv_l) / pv_l < 0.01`.

---

## Orthogonality matrix

| Cluster | Tasks |
|---------|-------|
| ALM / balance sheet | T01, T25 |
| Trading / microstructure | T02 |
| Alpha / factors | T03 |
| Quantum | T04, T05, T20 |
| Biology / medicine | T06, T07, T18, T19 |
| Physics simulation | T08, T14 |
| Engineering | T09, T22 |
| Control / optimization | T10, T23 |
| CS / crypto | T11, T12, T21 |
| Fluids / MD | T13, T14 |
| Quant finance | T15, T16 |
| ML / stats | T17, T24 |

## STEM10 cross-reference (prior batch)

| STEM10 | STEM25 thematic echo |
|--------|---------------------|
| T01 Black-Scholes | T15 min-var, T16 Heston |
| T02 VaR | T02 impact, T25 ALM PV |
| T03 Kalman | T24 continuous-discrete Kalman |
| T04 Schrödinger | T04 Heisenberg, T20 WKB |
| T05 density matrix | T05 Lindblad |
| T06 attention | T21 tensor contraction |
| T07 MCMC | T17 HMM (inference) |
| T08 Michaelis-Menten | T18 competitive inhibition |
| T09 rocket | T08 Lorenz-96 (dynamics) |
| T10 Ising | T14 LJ (stat mech) |

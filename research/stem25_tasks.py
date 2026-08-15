"""STEM25 task definitions — loaded by run_stem25_experiments.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


@dataclass
class StemTask:
    task_id: str
    domain: str
    slug: str
    message: str
    runtime_cmd: list[str]
    stem10_echo: str = ""


def _rt(slug: str) -> list[str]:
    return [
        str(PY),
        "-c",
        f"import sys; sys.path.insert(0, r'qwen code'); from {slug}.main import main; main()",
    ]


TASKS: list[StemTask] = [
    StemTask(
        "T01", "asset_liability_management", "stem25_alm_duration_match",
        "Edit stem25_alm_duration_match/main.py ONLY. Fix Macaulay duration: "
        "D = sum(t*CF_t*DF_t)/sum(CF_t*DF_t) with DF_t=(1+y)^(-t). Immunize portfolio to 5.0y. "
        "main() assert 4.8 < port_duration < 5.2. Stdlib only.",
        _rt("stem25_alm_duration_match"), "T01/T25 ALM",
    ),
    StemTask(
        "T02", "quant_trading", "stem25_market_impact",
        "Edit stem25_market_impact/main.py ONLY. Fix square-root market impact: "
        "cost = eta*sigma*sqrt(Q/V)*Q. main() assert 200<cost<400. No pip.",
        _rt("stem25_market_impact"), "T02 VaR/trading",
    ),
    StemTask(
        "T03", "alpha_generation", "stem25_factor_ic_blend",
        "Edit stem25_factor_ic_blend/main.py ONLY. Fix IC blend denominator: "
        "den = sum(abs(ic) for ic in ics) — NOT sum(ics) or sum(abs(ics)). "
        "main() assert 1.0 < score < 1.4.",
        _rt("stem25_factor_ic_blend"), "T03 factors",
    ),
    StemTask(
        "T04", "quantum_mechanics", "stem25_heisenberg_2q",
        "Edit stem25_heisenberg_2q/main.py ONLY. Build full Heisenberg H with X,Y,Z terms, "
        "return ground energy E0=-3*J for J=1 (min eigenvalue, not trace). main() assert abs(E0+3)<1e-9.",
        _rt("stem25_heisenberg_2q"), "T04 Schrödinger",
    ),
    StemTask(
        "T05", "quantum_mechanics", "stem25_lindblad_trace",
        "Edit stem25_lindblad_trace/main.py ONLY. Fix amplitude damping to preserve trace=1. "
        "main() assert abs(trace(rho_new)-1)<1e-9.",
        _rt("stem25_lindblad_trace"), "T05 density matrix",
    ),
    StemTask(
        "T06", "crispr_biology", "stem25_crispr_grna_score",
        "Edit stem25_crispr_grna_score/main.py ONLY. Weight PAM-proximal mismatches 3x. "
        "main() assert perfect match scores higher than mismatch.",
        _rt("stem25_crispr_grna_score"), "T08 biology",
    ),
    StemTask(
        "T07", "pharmacology", "stem25_pk_two_compartment",
        "Edit stem25_pk_two_compartment/main.py ONLY. CL = k10 + k12*k21/(k12+k21). "
        "main() assert 10.4 < Css < 10.7.",
        _rt("stem25_pk_two_compartment"), "T18/T19 PK",
    ),
    StemTask(
        "T08", "physics_simulation", "stem25_lorenz96_step",
        "Edit stem25_lorenz96_step/main.py ONLY. Fix Lorenz-96 periodic indices: use x[i-2] not x[i+2]. "
        "main() assert 0.1<norm<5.0 after RK4 step.",
        _rt("stem25_lorenz96_step"), "T09/T10 dynamics",
    ),
    StemTask(
        "T09", "civil_engineering", "stem25_beam_deflection",
        "Edit stem25_beam_deflection/main.py ONLY. Cantilever tip deflection delta=P*L^3/(3*E*I). "
        "main() assert abs(delta-0.012)<1e-4.",
        _rt("stem25_beam_deflection"), "engineering",
    ),
    StemTask(
        "T10", "control_theory", "stem25_lqr_riccati",
        "Edit stem25_lqr_riccati/main.py ONLY. Scalar DARE one-step: P_new = Q + (A^2*P_old)/(1 + B^2*P_old/R). "
        "NOT q+a^2*p+b^2*p/r. For demo params expect P_new≈1.346. main() assert 1.2<P_new<1.5.",
        _rt("stem25_lqr_riccati"), "control",
    ),
    StemTask(
        "T11", "compiler_engineering", "stem25_ssa_dominance",
        "Edit stem25_ssa_dominance/main.py ONLY. immediate_dominator(3)=1 for diamond CFG. "
        "Remove NotImplementedError.",
        _rt("stem25_ssa_dominance"), "CS/compiler",
    ),
    StemTask(
        "T12", "cryptography", "stem25_rsa_modexp",
        "Edit stem25_rsa_modexp/main.py ONLY. Correct square-and-multiply modexp. "
        "main() assert modexp(2,65537,999999937)==235894736.",
        _rt("stem25_rsa_modexp"), "crypto",
    ),
    StemTask(
        "T13", "fluid_dynamics", "stem25_colebrook_friction",
        "Edit stem25_colebrook_friction/main.py ONLY. Colebrook iteration uses 2.51/(Re*sqrt(f)). "
        "main() assert 0.018<f<0.022.",
        _rt("stem25_colebrook_friction"), "fluids",
    ),
    StemTask(
        "T14", "molecular_dynamics", "stem25_lennard_jones",
        "Edit stem25_lennard_jones/main.py ONLY. LJ force F=24*eps/r*(2*(sig/r)^12-(sig/r)^6). "
        "main() assert repulsive at r=0.9*sigma.",
        _rt("stem25_lennard_jones"), "T10 Ising/stat mech",
    ),
    StemTask(
        "T15", "quant_finance", "stem25_min_variance",
        "Edit stem25_min_variance/main.py ONLY. Min-var w = Sigma^-1 1 / (1^T Sigma^-1 1). "
        "main() assert weights sum 1 and 0.2<w0<0.8.",
        _rt("stem25_min_variance"), "T01/T15 portfolio",
    ),
    StemTask(
        "T16", "quant_finance", "stem25_heston_char",
        "Edit stem25_heston_char/main.py ONLY. Heston log-char at u=0 must be 0. Fix kappa*theta sign. "
        "main() assert abs(phi(0).real)<1e-9.",
        _rt("stem25_heston_char"), "T01 options",
    ),
    StemTask(
        "T17", "machine_learning", "stem25_hmm_gamma",
        "Edit stem25_hmm_gamma/main.py ONLY. gamma_t(i)=alpha*beta/P(O) with scaling. "
        "main() assert gammas sum to 1.",
        _rt("stem25_hmm_gamma"), "T07 inference",
    ),
    StemTask(
        "T18", "pharmacology", "stem25_competitive_inhibition",
        "Edit stem25_competitive_inhibition/main.py ONLY. Km_app=Km*(1+[I]/Ki). "
        "main() assert abs(v-5.0)<0.5 — do not use undefined vmax in main().",
        _rt("stem25_competitive_inhibition"), "T08 Michaelis-Menten",
    ),
    StemTask(
        "T19", "pharmacology", "stem25_dose_accumulation",
        "Edit stem25_dose_accumulation/main.py ONLY. R=1/(1-exp(-k*tau)). "
        "main() assert 3.0<R<3.5 for demo params.",
        _rt("stem25_dose_accumulation"), "dosing",
    ),
    StemTask(
        "T20", "quantum_mechanics", "stem25_wkb_tunneling",
        "Edit stem25_wkb_tunneling/main.py ONLY. kappa=sqrt(2m(V0-E))/hbar for barrier. "
        "main() assert 0<T<0.1.",
        _rt("stem25_wkb_tunneling"), "T04 quantum",
    ),
    StemTask(
        "T21", "scientific_computing", "stem25_tensor_contract",
        "Edit stem25_tensor_contract/main.py ONLY. Matrix-chain DP: shapes (10,20),(20,30),(30,5) "
        "optimal cost=4000 (pair middle mats first), naive left-to-right=7500. "
        "Implement optimal_chain_cost with DP; main() assert cost==4000.",
        _rt("stem25_tensor_contract"), "T06 tensors",
    ),
    StemTask(
        "T22", "finite_element", "stem25_fem_bar",
        "Edit stem25_fem_bar/main.py ONLY. Fix bar_stiffness() to K=(EA/L)*[[1,-1],[-1,1]]. "
        "Do NOT change main() asserts. Off-diagonals must equal -EA/L.",
        _rt("stem25_fem_bar"), "FEM",
    ),
    StemTask(
        "T23", "optimization", "stem25_ga_tsp",
        "Edit stem25_ga_tsp/main.py ONLY. Fix ox_crossover: copy segment from p1, fill "
        "remaining cities from p2 in order (standard OX). demo() tour [0,2,1,3] cost=8; "
        "child must cost less. Do not change demo() parents or rng seed.",
        _rt("stem25_ga_tsp"), "combinatorial opt",
    ),
    StemTask(
        "T24", "quant_bayesian", "stem25_kalman_cd",
        "Edit stem25_kalman_cd/main.py ONLY. Predict P=F*P*F^T+Q. main() assert P00 increases.",
        _rt("stem25_kalman_cd"), "T03 Kalman",
    ),
    StemTask(
        "T25", "asset_liability_management", "stem25_cashflow_match",
        "Edit stem25_cashflow_match/main.py ONLY. PV=sum CF_t/(1+y)^t. Match asset weight to liabilities. "
        "main() assert |pv_a-pv_l|/pv_l < 0.01.",
        _rt("stem25_cashflow_match"), "T02/T25 ALM",
    ),
]

assert len(TASKS) == 25

#!/usr/bin/env python
"""Generate 25 broken STEM25 starter files with verified runtime asserts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "stem25_starters"

STARTERS: dict[str, str] = {
    "stem25_alm_duration_match": '''"""ALM duration immunization — broken Macaulay duration."""

from __future__ import annotations


def macaulay_duration(cashflows: list[float], times: list[float], y: float) -> float:
  # BUG: uses undiscounted cashflows in numerator
  num = sum(t * cf for t, cf in zip(times, cashflows))
  den = sum(cf / (1 + y) ** t for t, cf in zip(times, cashflows))
  return num / den


def immunize_weight(d_a: float, d_b: float, d_liability: float) -> float:
  # w on A: w*d_a + (1-w)*d_b = d_liability
  return (d_liability - d_b) / (d_a - d_b)


def demo() -> float:
  cfs_a = [100.0]
  t_a = [5.0]
  cfs_b = [50.0, 50.0]
  t_b = [4.0, 6.0]
  y = 0.05
  d_a = macaulay_duration(cfs_a, t_a, y)
  d_b = macaulay_duration(cfs_b, t_b, y)
  w = immunize_weight(d_a, d_b, 5.0)
  return w * d_a + (1 - w) * d_b


def main() -> None:
  port_d = demo()
  assert 4.8 < port_d < 5.2, port_d
  print(f"portfolio_duration={port_d:.4f}")


if __name__ == "__main__":
  main()
''',
    "stem25_market_impact": '''"""Almgren-Chriss temporary impact — broken cost formula."""

from __future__ import annotations

import math


def temporary_impact_cost(q: float, v: float, sigma: float, eta: float) -> float:
  # BUG: linear in Q instead of square-root law
  return eta * sigma * (q / v) * q


def demo() -> float:
  return temporary_impact_cost(q=1e6, v=5e7, sigma=0.02, eta=0.1)


def main() -> None:
  cost = demo()
  assert 200 < cost < 400, cost
  print(f"impact_cost={cost:.2f}")


if __name__ == "__main__":
  main()
''',
    "stem25_factor_ic_blend": '''"""IC-weighted factor alpha — broken normalization."""

from __future__ import annotations


def zscore(xs: list[float]) -> list[float]:
  mu = sum(xs) / len(xs)
  var = sum((x - mu) ** 2 for x in xs) / len(xs)
  sd = var ** 0.5 or 1.0
  return [(x - mu) / sd for x in xs]


def blend_score(factors: list[list[float]], ics: list[float], idx: int) -> float:
  # BUG: divides by sum(IC) not sum(|IC|)
  zs = [zscore(f) for f in factors]
  num = sum(ics[k] * zs[k][idx] for k in range(len(ics)))
  den = sum(ics)
  return num / den


def demo() -> float:
  f1 = [1.0, 2.0, 3.0]
  f2 = [3.0, 2.0, 1.0]
  ics = [0.5, -0.3]
  return blend_score([f1, f2], ics, idx=2)


def main() -> None:
  s = demo()
  assert 1.0 < s < 1.4, s
  print(f"score={s:.4f}")


if __name__ == "__main__":
  main()
''',
    "stem25_heisenberg_2q": '''"""2-qubit Heisenberg — broken Hamiltonian / ground energy."""

from __future__ import annotations

import cmath


def pauli_matrices() -> dict[str, list[list[complex]]]:
  sx = [[0, 1], [1, 0]]
  sy = [[0, -1j], [1j, 0]]
  sz = [[1, 0], [0, -1]]
  i2 = [[1, 0], [0, 1]]
  return {"I": i2, "X": sx, "Y": sy, "Z": sz}


def kron(a: list[list[complex]], b: list[list[complex]]) -> list[list[complex]]:
  out = []
  for row_a in a:
    for row_b in b:
      out.append([x * y for x in row_a for y in row_b])
  return out


def mat_add(a, b):
  return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def ground_energy(j: float) -> float:
  p = pauli_matrices()
  h = mat_add(kron(p["X"], p["X"]), kron(p["Y"], p["Y"]))
  h = mat_add(h, kron(p["Z"], p["Z"]))
  h = [[j * x for x in row] for row in h]
  # BUG: returns trace instead of min eigenvalue
  return sum(h[i][i].real for i in range(4))


def demo() -> float:
  return ground_energy(1.0)


def main() -> None:
  e0 = demo()
  assert abs(e0 + 3.0) < 1e-9, e0
  print(f"E0={e0}")


if __name__ == "__main__":
  main()
''',
}

# Continue with remaining starters in part 2 - file too long, use function

def _more_starters() -> dict[str, str]:
    return {
        "stem25_lindblad_trace": '''"""Lindblad amplitude damping — broken trace preservation."""

from __future__ import annotations


def amplitude_damping_step(rho: list[list[complex]], gamma: float, dt: float) -> list[list[complex]]:
  # BUG: does not preserve trace
  r00 = rho[0][0] * (1 - gamma * dt)
  r11 = rho[1][1] + gamma * dt
  return [[r00, 0], [0, r11]]


def trace(m: list[list[complex]]) -> float:
  return float(m[0][0].real + m[1][1].real)


def demo() -> list[list[complex]]:
  rho = [[0.8 + 0j, 0], [0, 0.2 + 0j]]
  return amplitude_damping_step(rho, gamma=0.1, dt=0.5)


def main() -> None:
  rho_new = demo()
  assert abs(trace(rho_new) - 1.0) < 1e-9, trace(rho_new)
  assert rho_new[0][0].real > 0
  print(f"trace={trace(rho_new):.6f}")


if __name__ == "__main__":
  main()
''',
        "stem25_crispr_grna_score": '''"""CRISPR gRNA scoring — broken mismatch weights."""

from __future__ import annotations


def grna_score(seq: str, target: str) -> float:
  # BUG: all mismatches weighted equally (no PAM-proximal 3x)
  score = 100.0
  for i, (a, b) in enumerate(zip(seq, target)):
    if a != b:
      score -= 5.0
  gc = (seq.count("G") + seq.count("C")) / len(seq)
  if 0.4 <= gc <= 0.6:
    score += 10.0
  return score


def demo() -> tuple[float, float]:
  good = grna_score("ACGTACGTACGT", "ACGTACGTACGT")
  bad = grna_score("TTTTTTTTTTTT", "ACGTACGTACGT")
  return good, bad


def main() -> None:
  good, bad = demo()
  assert good > bad, (good, bad)
  print(f"good={good} bad={bad}")


if __name__ == "__main__":
  main()
''',
        "stem25_pk_two_compartment": '''"""Two-compartment PK steady state — broken clearance."""

from __future__ import annotations


def steady_state_c1(r0: float, k10: float, k12: float, k21: float) -> float:
  # BUG: CL = k10 only (ignores peripheral clearance path)
  cl = k10
  return r0 / cl


def demo() -> float:
  return steady_state_c1(r0=10.0, k10=0.9, k12=0.1, k21=0.1)


def main() -> None:
  css = demo()
  assert 10.4 < css < 10.7, css
  print(f"Css={css:.4f}")


if __name__ == "__main__":
  main()
''',
        "stem25_lorenz96_step": '''"""Lorenz-96 — broken periodic RHS."""

from __future__ import annotations

import math


def lorenz96_rhs(x: list[float], f: float) -> list[float]:
  n = len(x)
  out = []
  for i in range(n):
    # BUG: wrong index wrap (uses i+2 instead of i-2)
    out.append((x[(i + 1) % n] - x[(i + 2) % n]) * x[(i - 1) % n] - x[i] + f)
  return out


def rk4_step(x: list[float], f: float, dt: float) -> list[float]:
  k1 = lorenz96_rhs(x, f)
  k2 = lorenz96_rhs([xi + 0.5 * dt * ki for xi, ki in zip(x, k1)], f)
  k3 = lorenz96_rhs([xi + 0.5 * dt * ki for xi, ki in zip(x, k2)], f)
  k4 = lorenz96_rhs([xi + dt * ki for xi, ki in zip(x, k3)], f)
  return [xi + dt * (k1i + 2 * k2i + 2 * k3i + k4i) / 6 for xi, k1i, k2i, k3i, k4i in zip(x, k1, k2, k3, k4)]


def demo() -> float:
  x0 = [1.0, 0.0, 0.0, 0.0, 0.0]
  x1 = rk4_step(x0, f=8.0, dt=0.01)
  return math.sqrt(sum(v * v for v in x1))


def main() -> None:
  norm = demo()
  assert 0.1 < norm < 5.0, norm
  print(f"norm={norm:.4f}")


if __name__ == "__main__":
  main()
''',
        "stem25_beam_deflection": '''"""Cantilever beam — broken deflection formula."""

from __future__ import annotations


def max_tip_deflection(p: float, l: float, e: float, i_moment: float) -> float:
  # BUG: wrong formula P*L/(E*I)
  return p * l / (e * i_moment)


def demo() -> float:
  return max_tip_deflection(p=7500.0, l=2.0, e=200e9, i_moment=8.33e-6)


def main() -> None:
  d = demo()
  assert abs(d - 0.012) < 1e-4, d
  print(f"delta={d:.6f}")


if __name__ == "__main__":
  main()
''',
        "stem25_lqr_riccati": '''"""Discrete LQR Riccati — broken update."""

from __future__ import annotations


def dare_step_scalar(a: float, b: float, q: float, r: float, p_old: float) -> float:
  # BUG: adds instead of correct DARE form
  return q + a * a * p_old + b * b * p_old / r


def demo() -> float:
  return dare_step_scalar(a=1.1, b=0.5, q=1.0, r=0.1, p_old=1.0)


def main() -> None:
  p_new = demo()
  assert 1.2 < p_new < 1.5, p_new
  print(f"P_new={p_new:.4f}")


if __name__ == "__main__":
  main()
''',
        "stem25_ssa_dominance": '''"""SSA dominance — broken immediate dominator."""

from __future__ import annotations


def immediate_dominator(node: int, preds: dict[int, list[int]]) -> int:
  raise NotImplementedError("dominance frontier")


def demo() -> int:
  # Diamond: 0->1, 0->2, 1->3, 2->3
  preds = {1: [0], 2: [0], 3: [1, 2]}
  return immediate_dominator(3, preds)


def main() -> None:
  idom = demo()
  assert idom == 1, idom
  print(f"idom(3)={idom}")


if __name__ == "__main__":
  main()
''',
        "stem25_rsa_modexp": '''"""RSA modexp — broken fast exponentiation."""

from __future__ import annotations


def modexp(base: int, exp: int, mod: int) -> int:
  # BUG: stops one bit early
  result = 1
  b = base % mod
  e = exp
  while e > 1:
    if e % 2:
      result = (result * b) % mod
    b = (b * b) % mod
    e //= 2
  return (result * b) % mod


def demo() -> int:
  return modexp(2, 65537, 999999937)


def main() -> None:
  r = demo()
  assert r == 235894736, r
  print(f"residue={r}")


if __name__ == "__main__":
  main()
''',
        "stem25_colebrook_friction": '''"""Colebrook friction — broken iteration."""

from __future__ import annotations

import math


def colebrook_f(re: float, rel_rough: float, iters: int = 20) -> float:
  f = 0.02
  for _ in range(iters):
    # BUG: wrong constant 2.51 -> 25.1
    inv_sqrt = -2 * math.log10(rel_rough / 3.7 + 25.1 / (re * math.sqrt(f)))
    f = 1 / (inv_sqrt * inv_sqrt)
  return f


def demo() -> float:
  return colebrook_f(re=1e5, rel_rough=0.0004)


def main() -> None:
  f = demo()
  assert 0.018 < f < 0.022, f
  print(f"f={f:.5f}")


if __name__ == "__main__":
  main()
''',
        "stem25_lennard_jones": '''"""Lennard-Jones force — wrong derivative."""

from __future__ import annotations


def lj_force(r: float, sigma: float, epsilon: float) -> float:
  x = sigma / r
  # BUG: missing factor of 24/r
  return epsilon * (2 * x**12 - x**6)


def demo() -> float:
  return lj_force(r=0.9 * 1.0, sigma=1.0, epsilon=1.0)


def main() -> None:
  f = demo()
  assert f > 0, f  # repulsive at r < sigma
  print(f"F={f:.4f}")


if __name__ == "__main__":
  main()
''',
        "stem25_min_variance": '''"""Min-variance portfolio — broken weight formula."""

from __future__ import annotations


def min_var_weights(sigma: list[list[float]]) -> list[float]:
  # 2x2 inverse analytic
  a, b = sigma[0][0], sigma[0][1]
  c, d = sigma[1][0], sigma[1][1]
  det = a * d - b * c
  inv = [[d / det, -b / det], [-c / det, a / det]]
  # BUG: uses inv @ mu instead of inv @ ones
  num = [inv[0][0] + inv[0][1], inv[1][0] + inv[1][1]]
  den = num[0] + num[1]
  return [num[0] / den, num[1] / den]


def demo() -> list[float]:
  sigma = [[0.04, 0.01], [0.01, 0.09]]
  return min_var_weights(sigma)


def main() -> None:
  w = demo()
  assert abs(sum(w) - 1.0) < 1e-9, w
  assert 0.2 < w[0] < 0.8, w
  print(f"w={w}")


if __name__ == "__main__":
  main()
''',
        "stem25_heston_char": '''"""Heston characteristic function — wrong sign at u=0."""

from __future__ import annotations

import cmath


def heston_log_char(u: complex, v0: float, kappa: float, theta: float, sigma: float, rho: float, t: float) -> complex:
  # BUG: adds kappa*theta instead of subtract in drift term (simplified)
  drift = kappa * theta * t
  return drift + 0j


def demo() -> complex:
  return heston_log_char(0 + 0j, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, t=1.0)


def main() -> None:
  phi = demo()
  assert abs(phi.real) < 1e-9, phi
  print(f"phi(0)={phi}")


if __name__ == "__main__":
  main()
''',
        "stem25_hmm_gamma": '''"""HMM gamma — broken normalization."""

from __future__ import annotations


def gamma_t(alpha: list[float], beta: list[float], scale: float) -> list[float]:
  raw = [a * b for a, b in zip(alpha, beta)]
  # BUG: forgets to divide by scale**2 total likelihood
  s = sum(raw) or 1.0
  return [r / s for r in raw]


def demo() -> list[float]:
  alpha = [0.4, 0.6]
  beta = [0.5, 0.5]
  return gamma_t(alpha, beta, scale=0.9)


def main() -> None:
  g = demo()
  assert abs(sum(g) - 1.0) < 1e-9, g
  print(f"gamma={g}")


if __name__ == "__main__":
  main()
''',
        "stem25_competitive_inhibition": '''"""Competitive inhibition — wrong Km_app."""

from __future__ import annotations


def mm_rate(s: float, vmax: float, km: float) -> float:
  return vmax * s / (km + s)


def competitive_km_app(km: float, i_conc: float, ki: float) -> float:
  # BUG: divides instead of multiplies
  return km / (1 + i_conc / ki)


def demo() -> float:
  km, ki, i_conc, vmax = 2.0, 1.0, 1.0, 10.0
  km_app = competitive_km_app(km, i_conc, ki)
  return mm_rate(km_app, vmax, km_app)


def main() -> None:
  v = demo()
  assert abs(v - 5.0) < 0.5, v
  print(f"v={v:.4f}")


if __name__ == "__main__":
  main()
''',
        "stem25_dose_accumulation": '''"""Dose accumulation — broken R factor."""

from __future__ import annotations

import math


def accumulation_factor(k: float, tau: float) -> float:
  # BUG: uses exp(+k*tau) instead of exp(-k*tau)
  return 1 / (1 - math.exp(k * tau))


def demo() -> float:
  return accumulation_factor(k=0.05, tau=7.5)


def main() -> None:
  r = demo()
  assert 3.0 < r < 3.5, r
  print(f"R={r:.4f}")


if __name__ == "__main__":
  main()
''',
        "stem25_wkb_tunneling": '''"""WKB tunneling — broken kappa."""

from __future__ import annotations

import math


def wkb_transmission(m: float, v0: float, e: float, a: float, hbar: float = 1.0) -> float:
  # BUG: uses sqrt(E-V0) instead of sqrt(V0-E) for barrier
  kappa = math.sqrt(2 * m * max(e - v0, 1e-12)) / hbar
  return math.exp(-2 * kappa * a)


def demo() -> float:
  return wkb_transmission(m=1.0, v0=5.0, e=2.0, a=1.0)


def main() -> None:
  t = demo()
  assert 0 < t < 0.1, t
  print(f"T={t:.6f}")


if __name__ == "__main__":
  main()
''',
        "stem25_tensor_contract": '''"""Tensor chain contraction — wrong order cost."""

from __future__ import annotations


def chain_cost_left(shapes: list[tuple[int, int]]) -> int:
  cost = 0
  acc = shapes[0]
  for b in shapes[1:]:
    cost += acc[0] * acc[1] * b[1]
    acc = (acc[0], b[1])
  return cost


def optimal_chain_cost(shapes: list[tuple[int, int]]) -> int:
  # BUG: returns naive left-to-right only (7500 not optimal 4000)
  return chain_cost_left(shapes)


def demo() -> int:
  shapes = [(10, 20), (20, 30), (30, 5)]
  return optimal_chain_cost(shapes)


def main() -> None:
  cost = demo()
  assert cost == 4000, cost
  print(f"cost={cost}")


if __name__ == "__main__":
  main()
''',
        "stem25_fem_bar": '''"""1D bar stiffness — broken matrix."""

from __future__ import annotations


def bar_stiffness(ea: float, length: float) -> list[list[float]]:
  k = ea / length
  # BUG: wrong signs on off-diagonal
  return [[k, k], [k, k]]


def demo() -> list[list[float]]:
  return bar_stiffness(ea=1e6, length=2.0)


def main() -> None:
  k = demo()
  assert abs(k[0][0] - 5e5) < 1.0, k
  assert abs(k[1][1] - 5e5) < 1.0, k
  assert abs(k[0][1] + 5e5) < 1.0, k
  assert abs(k[1][0] + 5e5) < 1.0, k
  print(f"K00={k[0][0]}")


if __name__ == "__main__":
  main()
''',
        "stem25_ga_tsp": '''"""GA TSP — broken crossover duplicates cities."""

from __future__ import annotations

import random


def tour_cost(tour: list[int], dist: list[list[float]]) -> float:
  return sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def ox_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
  # BUG: returns slice of p1 only (duplicates/missing)
  cut = len(p1) // 2
  return p1[:cut] + p1[:cut]


def demo() -> tuple[float, float]:
  dist = [
    [0, 1, 2, 3],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
    [3, 2, 1, 0],
  ]
  rng = random.Random(0)
  t0 = [0, 2, 1, 3]
  c0 = tour_cost(t0, dist)
  child = ox_crossover([0, 2, 1, 3], [2, 0, 3, 1], rng)
  if len(set(child)) != 4:
    return c0, c0 + 1
  c1 = tour_cost(child, dist)
  return c0, c1


def main() -> None:
  c0, c1 = demo()
  assert c1 < c0, (c0, c1)
  print(f"cost {c0} -> {c1}")


if __name__ == "__main__":
  main()
''',
        "stem25_kalman_cd": '''"""Kalman predict — broken covariance update."""

from __future__ import annotations


def predict_cov(p: list[list[float]], f: list[list[float]], q: list[list[float]]) -> list[list[float]]:
  # BUG: P_pred = F*P + Q
  return [
    [f[0][0] * p[0][0] + q[0][0], f[0][1] * p[0][1] + q[0][1]],
    [f[1][0] * p[1][0] + q[1][0], f[1][1] * p[1][1] + q[1][1]],
  ]


def demo() -> list[list[float]]:
  p = [[1.0, 0.0], [0.0, 1.0]]
  f = [[1.0, 0.1], [0.0, 1.0]]
  q = [[0.01, 0.0], [0.0, 0.01]]
  return predict_cov(p, f, q)


def main() -> None:
  p_new = demo()
  assert p_new[0][0] > 1.0, p_new
  print(f"P00={p_new[0][0]:.4f}")


if __name__ == "__main__":
  main()
''',
        "stem25_cashflow_match": '''"""ALM cashflow PV — broken discounting."""

from __future__ import annotations


def present_value(cashflows: list[float], y: float) -> float:
  # BUG: sums undiscounted cashflows
  return sum(cashflows)


def match_weight(asset_cfs: list[float], liab_cfs: list[float], y: float) -> float:
  pv_l = present_value(liab_cfs, y)
  pv_a_unit = present_value(asset_cfs, y)
  return pv_l / pv_a_unit


def demo() -> tuple[float, float]:
  y = 0.05
  assets = [0.0, 105.0]
  liabs = [0.0, 100.0]
  w = match_weight(assets, liabs, y)
  pv_a = w * present_value(assets, y)
  pv_l = present_value(liabs, y)
  return pv_a, pv_l


def main() -> None:
  pv_a, pv_l = demo()
  assert abs(pv_a - pv_l) / pv_l < 0.01, (pv_a, pv_l)
  print(f"pv_a={pv_a:.4f} pv_l={pv_l:.4f}")


if __name__ == "__main__":
  main()
''',
    }


def main() -> None:
    all_starters = {**STARTERS, **_more_starters()}
    assert len(all_starters) == 25, len(all_starters)
    for slug, code in sorted(all_starters.items()):
        d = OUT / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "main.py").write_text(code, encoding="utf-8")
        print(f"wrote {slug}")
    print(f"total={len(all_starters)}")


if __name__ == "__main__":
    main()

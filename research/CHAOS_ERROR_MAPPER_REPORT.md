# Chaos Error Mapper — Cross-Field Report

**Invocations profiled:** 135
**Experiments:** 46

## Top recipes (all fields)

| Recipe | Fires |
|--------|-------|
| F_LENS | 2456 |
| A_UNCERTAIN | 198 |
| F_E_DEPTH | 21 |
| E_CLIFF | 21 |
| B_STUB | 12 |
| B_STUB_DISTRESS | 4 |
| B_DISTRESS | 3 |
| C_OSC | 2 |
| AGENT_FLAIL | 2 |
| C_F_TRAJ | 1 |

## Top atoms

| Atom | Steps |
|------|-------|
| F_disagree | 2347 |
| A_uncertain | 196 |
| B_stub | 17 |
| E_cliff | 13 |
| B_distress | 8 |
| C_osc | 3 |
| D_break | 3 |
| G_oob | 3 |

## Top fine flags (token hits)

- `B_low_code`: 1420
- `A_low_margin`: 212
- `B_stub`: 21
- `B_distress`: 12
- `C_oscillation`: 9
- `C_thrash`: 7
- `A_high_entropy`: 1

## Checker fires (A–G)

- **F**: 130 invocations
- **B_transport**: 124 invocations
- **B_jlens**: 124 invocations
- **E**: 124 invocations
- **A**: 20 invocations
- **B_vocab**: 12 invocations
- **G**: 10 invocations
- **C**: 1 invocations

## Top error signatures → CAA hints

### `F_disagree+F_LENS(2)` (35 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T01 / quant_finance / RESOLVED stub=0
  - T02 / quant_finance / RESOLVED stub=0
  - T03 / quant_bayesian / RESOLVED stub=0
  - T04 / quantum_mechanics / RESOLVED stub=0
  - T05 / quantum_mechanics / RESOLVED stub=0

### `F_disagree+F_LENS(3)` (19 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T01 / quant_finance / RESOLVED stub=0
  - T06 / ai_development / RESOLVED stub=0
  - T07 / bayesian_statistics / RESOLVED stub=0
  - T01 / asset_liability_management / RESOLVED stub=0
  - T07 / pharmacology / RESOLVED stub=0

### `A_uncertain+F_disagree+F_LENS(3)+A_UNCERTAIN(1)` (8 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T02 / quant_finance / RESOLVED stub=0
  - T08 / unknown / UNKNOWN stub=0
  - T09 / rocket_science / RESOLVED stub=0
  - T04 / quantum_mechanics / RESOLVED stub=0
  - T10 / control_theory / RESOLVED stub=0

### `A_uncertain+F_disagree+F_LENS(2)+A_UNCERTAIN(1)` (6 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T03 / quant_bayesian / RESOLVED stub=0
  - T10 / statistical_physics / PARTIAL stub=2
  - T05 / quantum_mechanics / RESOLVED stub=0
  - T12 / cryptography / RESOLVED stub=0
  - T16 / quant_finance / RESOLVED stub=0

### `F_disagree+F_LENS(4)` (3 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T04 / quantum_mechanics / RESOLVED stub=0
  - T08 / unknown / UNKNOWN stub=0
  - T09 / rocket_science / RESOLVED stub=0

### `A_uncertain+C_osc+D_break+E_cliff+F_disagree+G_oob` (3 inv)
- **CAA hint:** UNI_CAA04 / UNI_CAA05
  - H3_PANIC_DEMO / live_panic_demo_scheduler / RESOLVED stub=0
  - H3_PANIC_DEMO / live_panic_demo_scheduler / RESOLVED stub=0
  - H3_PANIC_DEMO / live_panic_demo_scheduler / RESOLVED stub=0

### `A_uncertain+F_disagree+F_LENS(1)+A_UNCERTAIN(1)` (2 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T03 / alpha_generation / RESOLVED stub=0
  - live_panic_demo_scheduler / unknown / UNKNOWN stub=0

### `A_uncertain+F_disagree+F_LENS(81)+A_UNCERTAIN(7)` (2 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T13 / fluid_dynamics / RESOLVED stub=0
  - T21 / scientific_computing / RESOLVED stub=0

### `A_uncertain+F_disagree+F_LENS(41)+A_UNCERTAIN(4)` (2 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T13 / fluid_dynamics / RESOLVED stub=0
  - T21 / scientific_computing / RESOLVED stub=0

### `A_uncertain+F_disagree+F_LENS(2)+A_UNCERTAIN(2)` (2 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - T18 / pharmacology / RESOLVED stub=0
  - T23 / optimization / RESOLVED stub=0

### `E_cliff+F_disagree+F_LENS(2)` (2 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - R2_STUB / research_stub_queue / RESOLVED stub=0
  - R3_DISTRESS / research_distress_bucket / RESOLVED stub=0

### `A_uncertain+E_cliff+F_disagree` (2 inv)
- **CAA hint:** UNI_CAA08 lens-corroborated (gate with B_*)
  - H1_CALIB_STRESS / live_panic_demo_scheduler / RESOLVED stub=0
  - H1_CALIB_STRESS / live_panic_demo_scheduler / RESOLVED stub=0


## By domain (recipe fires)

**ai_development:** F_LENS(65), A_UNCERTAIN(2)
**alpha_generation:** F_LENS(36), A_UNCERTAIN(3)
**asset_liability_management:** F_LENS(75), A_UNCERTAIN(10)
**bayesian_statistics:** F_LENS(71), A_UNCERTAIN(8)
**chaos_proof_clean:** F_LENS(23), A_UNCERTAIN(3)
**civil_engineering:** F_LENS(40), A_UNCERTAIN(3)
**compiler_engineering:** F_LENS(49), A_UNCERTAIN(1)
**control_theory:** F_LENS(38), A_UNCERTAIN(3)
**crispr_biology:** F_LENS(59), A_UNCERTAIN(6), C_OSC(2), C_F_TRAJ(1)
**cryptography:** F_LENS(28), A_UNCERTAIN(6)
**finite_element:** F_LENS(41), A_UNCERTAIN(3)
**fluid_dynamics:** F_LENS(132), A_UNCERTAIN(11)
**machine_learning:** F_LENS(42), A_UNCERTAIN(5)
**molecular_dynamics:** F_LENS(39), A_UNCERTAIN(2)
**optimization:** F_LENS(64), A_UNCERTAIN(6)
**pharmacology:** F_LENS(125), A_UNCERTAIN(7)
**physics_simulation:** F_LENS(77), A_UNCERTAIN(2)
**quant_bayesian:** F_LENS(129), A_UNCERTAIN(12), B_STUB(1)
**quant_finance:** F_LENS(285), A_UNCERTAIN(19), B_STUB(2)
**quant_trading:** F_LENS(36), A_UNCERTAIN(4)
**quantum_mechanics:** F_LENS(313), A_UNCERTAIN(17), B_STUB(2), B_DISTRESS(2)
**research_distress_bucket:** F_LENS(51), A_UNCERTAIN(3)
**research_stub_queue:** F_LENS(48), A_UNCERTAIN(3)
**rocket_science:** F_LENS(60), A_UNCERTAIN(2)
**scientific_computing:** F_LENS(132), A_UNCERTAIN(11)
**statistical_physics:** F_LENS(75), A_UNCERTAIN(7), B_STUB(2)
**unknown:** F_LENS(323), A_UNCERTAIN(39), F_E_DEPTH(21), E_CLIFF(21), B_STUB(5)

# CAA Natural Mining & Neutral Extraction — Data Inventory

**Workspace:** `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It`  
**Gate snapshot:** `n_natural_admitted=0`, `production_ready=false`, `pipeline_ready=true` (`artifacts/gates/caa_steering_status.json`)

---

## 1. Neutral-related

### `NEUTRAL_TASKS` (`src/jlens_qwen/caa/neutral_capture.py`)

| slug | user_prompt (summary) | assistant_start |
|------|----------------------|-----------------|
| `chaos_proof_clean` | Edit fibonacci(10)→55, lint-clean | `` ```python\nimport `` |
| `chaos_proof_meta_wipe` | Fix `is_palindrome('racecar')`, lint-clean, never pip | `` ```python\ndef is_palindrome `` |

Requires ≥2 tasks (N-S2). Live capture via `research/caa_extract_neutral.py` with `CAA_NEUTRAL_LIVE=1` → writes `neutral_L{L}.pt` + `neutral_capture.json`.

### Neutral artifacts under `artifacts/caa/UNI_CAA01/`

| Path | Exists? | Usable? |
|------|---------|---------|
| `neutral_L10.pt`, `neutral_L16.pt`, `neutral_L22.pt` | **No** (0 files repo-wide) | — |
| `neutral_capture.json` | **No** | — |
| Any `neutral*` under UNI_CAA01 | **No** (only source module) | — |

**Live neutral not run.** Gate is unit-only:

- `artifacts/gates/caa_neutral_purity_gate.json` — `pass: true`, `environment.mode: "unit"`
- N1 text scoring passes for both tasks; N3–N6 deferred to live

### Prior `chaos_proof_clean` / `chaos_proof_meta_wipe` runs (`experiments/out`)

| File | Stamp | Device | n_inv | token_ids | eligible | d_consensus |
|------|-------|--------|-------|-----------|----------|-------------|
| `aider_chaos_proof_clean_latest.json` | 20260806T213555Z | xpu | 1 | ✅ v2 chaos, all steps | ❌ | null |
| `aider_chaos_proof_clean_20260806T213555Z.json` | (dup of latest) | xpu | 1 | ✅ | ❌ | null |
| `aider_chaos_proof_clean_20260805T094909Z.json` | 20260805T094909Z | — | 1 | ✅ | ❌ | null |
| `aider_chaos_proof_meta_wipe_latest.json` | 20260806T214152Z | xpu | 1 | ✅ | ❌ | null |
| `aider_chaos_proof_meta_wipe_20260806T214152Z.json` | (dup) | xpu | 1 | ✅ | ❌ | null |
| `aider_chaos_proof_meta_wipe_20260805T095621Z.json` | 20260805T095621Z | — | 1 | ✅ | ❌ | null |

Sessions: `experiments/out/aider_sessions/aider_chaos_proof_clean_20260806T213417Z.jsonl`, `...meta_wipe_20260806T214036Z.jsonl`

**Note:** `chaos_proof_lint_stress` has 3 XPU runs but is **not** in `NEUTRAL_TASKS`.

---

## 2. Panic / divergence corpus

### `divergence_*_latest.json` (4 files)

| File | Task | n_inv | token_ids | eligible | d_consensus | B_STUB_DISTRESS (timeline) |
|------|------|-------|-----------|----------|-------------|---------------------------|
| `experiments/out/divergence_corpus_stress_b_latest.json` | `live_panic_demo_scheduler` | 1 | ✅ 485 tokens | ✅ high | **55** | ✅ step 55 |
| `experiments/out/divergence_corpus_scheduler_latest.json` | `live_panic_demo_scheduler` | 2 | ✅ 504+21 | ❌ both | null | ❌ |
| `experiments/out/divergence_corpus_meta_wipe_latest.json` | `chaos_proof_meta_wipe` | 1 | ✅ 209 | ❌ | null | ❌ |
| `experiments/out/divergence_live_proof_latest.json` | `chaos_proof_clean` | 1 | ✅ 164 | ❌ | null | ❌ |

Timestamped twins (same content class):  
`divergence_corpus_stress_b_20260809T193806Z.json`, `divergence_corpus_scheduler_20260809T193415Z.json`, `divergence_corpus_meta_wipe_20260809T194029Z.json`, `divergence_live_proof_20260809T190650Z.json`, `divergence_live_proof_20260809T190001Z.json`

Sessions (5): `experiments/out/aider_sessions/divergence_corpus_{meta_wipe,scheduler,stress_b}_*.jsonl`, `divergence_live_proof_*.jsonl`

### Corpus-wide panic eligibility counts

Scanned all `experiments/out/*.json` with `prefix_divergence_tracer.eligible_for_panic_trace`:

| Metric | Unique invocations | File entries (incl. latest+stamped dupes) |
|--------|-------------------|-------------------------------------------|
| **Eligible (panic trace)** | **3** | 6 |
| **B_STUB_DISTRESS in `monitor.recipe_timeline`** | **3** | 6 |
| **With `d_consensus`** | **3** | 6 |

**3 unique eligible invocations:**

| Source file | Experiment | Task | d_consensus | confidence | prefix_token_ids |
|-------------|------------|------|-------------|------------|------------------|
| `divergence_corpus_stress_b_latest.json` | `divergence_corpus_stress_b` | `live_panic_demo_scheduler` | 55 | high | len=55 ✅ |
| `monitor_gate_stress_latest.json` | `monitor_gate_stress` | `live_panic_demo_scheduler` | 40 | medium | len=21 ✅ |
| `stem10_t08_stem10_michaelis_menten_latest.json` | `stem10_t08` | `stem10_michaelis_menten` | 69 | medium | len=36 ✅ |

`divergence_corpus_stress_b` audited in `artifacts/gates/divergence_metrics_audit.json` — all checks pass (token_ids_len=516, prefix_token_ids match d=55).

**Mapper aggregate** (`research/results/chaos_mapper_report.json`): 3 profiles with `"eligible": true`; `research/CHAOS_ERROR_MAPPER_REPORT.md` reports 4 `B_STUB_DISTRESS` recipe fires across the broader mapper corpus (7 timeline hits in JSON).

**Non-eligible runs still have usable chaos v2 `token_ids`** (e.g. scheduler, meta_wipe, live_proof) — just no panic `d_consensus`.

### `prefix_divergence_tracer` usage

| Consumer | Role |
|----------|------|
| `research/prefix_divergence_tracer.py` | Core tracer (`trace_invocation`, `eligible_for_panic_trace`, `trace_experiment`) |
| `research/validate_caa_pair.py` | Pair T-gates (eligible, B_STUB_DISTRESS) |
| `research/caa_pair_practice.py` | Practice pair capture |
| `research/audit_divergence_metrics.py` | Metrics audit → `artifacts/gates/divergence_metrics_audit.json` |
| `research/validate_divergence_tracer.py` | Offline/live proof → `artifacts/gates/divergence_tracer_proof.json` |
| `research/chaos_error_mapper.py` | Mapper divergence profiles |
| `research/extend_divergence_corpus.py` | Grows corpus via 3 stress Aider runs |
| `tests/unit/test_prefix_divergence_tracer.py` | Unit tests |

### `artifacts/caa/UNI_CAA01/pairs/`

**Does not exist.** No natural pair captures on disk.

### `practice_pair/` contents

`c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\artifacts\caa\UNI_CAA01\practice_pair\`

| File | Size / notes | Usable fields |
|------|--------------|---------------|
| `capture_manifest.json` | `source=practice`, `pool_eligible=false`, `kv_live_gate_pass=false`, d=77 | Plumbing only |
| `admission_gate.json` | `pass=true`, `pool_eligible=false` | Never pooled |
| `practice_pair_gate.json` | `pass=true`, d=77 | KV fixture |
| `matched_prefix_spec.json` | d=77, **prefix_token_ids** (77 ints) ✅ | Real prefix lock |
| `activations_at_d.json` | ~43K lines, L10/16/22 activations | Shared-KV fixture data |
| `chaos_summary_calm.json` | eligible=false, d=null, n_tokens=174 | Clean calm replay |
| `chaos_summary_bad.json` | eligible=false, d=null; B_DISTRESS@86, B_STUB@95 (separate, not combo) | Near-panic but **not** tracer-eligible |
| `practice_pair_result.json` | Full XPU practice run | Both arms eligible=false |

**No `activations_at_d.pt`** anywhere under `artifacts/caa/` (only JSON for practice; fixture manifest hashes a `.pt` that is **not on disk**).

### `calm_tails/` status

**Directory does not exist.**  
Blocked by `caa_mine_natural_pairs.py --live` and `artifacts/gates/caa_natural_pair_live_gate.json` (`pass: false`, blocker: *"No curated calm tails under artifacts/caa/UNI_CAA01/calm_tails/"*).  
Expected: manually curated `*.txt` prefix-locked calm completions per `research/notes/miner_gaps.md`.

### Fixture synth (`fixtures/natural_pair_synth/`)

6 JSON files only — synthetic token IDs (100–102 prefix), not real model tokens:

- `method_b_calm.json` / `method_b_bad.json` — d=3, synthetic `prefix_token_ids`
- `purity_gate.json` — calm eligible=false; bad eligible=true (fixture)
- `admission_gate.json` — `pool_eligible=false`, `kv_live_gate_pass=false`
- **No real activations on disk** (manifest references `activations_at_d.pt` hash but file absent)

---

## 3. HAVE vs NEED (≥3 admitted natural pairs + live neutral)

### HAVE

| Asset | Count | Status |
|-------|-------|--------|
| Admitted natural pairs (`pairs/natural/`) | **0** | Blocked |
| Practice pair (KV plumbing) | **1** | `pool_eligible=false` forever |
| Fixture synth pair | **1** | Unit test only, not poolable |
| Aggregated steering vectors | **6** `.pt` (behavior/condition L10/16/22) | Bootstrap from practice, `n_pairs=1` |
| Tracer-eligible panic invocations | **3** unique | Full `token_ids` + `d_consensus` |
| Divergence corpus runs (extend script) | **4** latest experiments | 1 eligible, 3 clean/control |
| Neutral unit gate | **pass** | Text purity only |
| `chaos_proof_clean` + `meta_wipe` XPU runs | **2+2** timestamps | Usable chaos v2, not panic-eligible |
| Phase 3A admission gate | **pass** (fixture) | |
| Phase 3B live miner | **stub** | Exits on missing prerequisites |
| Calm-tail purity gate | **pass** (pytest) | No real tails to test |

### NEED (gaps)

| Requirement | Gap |
|-------------|-----|
| **≥3 admitted natural pairs** | Need `calm_tails/` curation → Phase 3B miner loop → KV live capture → write `pairs/natural/{slug}/` with `pool_eligible=true` + `kv_live_gate_pass=true` |
| **Live neutral (`neutral_L*.pt`)** | Run `CAA_NEUTRAL_LIVE=1 python research/caa_extract_neutral.py` on XPU |
| **Calm tails per eligible bad arm** | Manual curation under `artifacts/caa/UNI_CAA01/calm_tails/` — **0 files today** |
| **Pair mining from corpus** | 3 eligible invocations are raw source material; none paired/mined/admitted |
| **Production steering** | `n_pairs=1` bootstrap only; need ≥3 natural for `production_ready` |
| **n_corpus_eligible in master gate** | Still `null` |

### Readiness matrix

```
Component                    HAVE                          NEED
─────────────────────────────────────────────────────────────────
Panic corpus (tracer-ready)  3 invocations w/ d_consensus  Calm tail per bad arm
Calm tails directory         MISSING                       ≥1 .txt per candidate pair
pairs/natural/               MISSING (0)                   ≥3 admitted captures
neutral_L*.pt                MISSING (0)                     Live XPU extract
Phase 3B miner               Blocked stub                  Implement + run
pool_eligible natural        0                             ≥3
```

---

## Quick reference — key absolute paths

**Eligible panic sources (usable for mining input):**
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\experiments\out\divergence_corpus_stress_b_latest.json`
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\experiments\out\monitor_gate_stress_latest.json`
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\experiments\out\stem10_t08_stem10_michaelis_menten_latest.json`

**Neutral task runs (not yet extracted):**
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\experiments\out\aider_chaos_proof_clean_latest.json`
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\experiments\out\aider_chaos_proof_meta_wipe_latest.json`

**Existing CAA artifacts (practice bootstrap only):**
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\artifacts\caa\UNI_CAA01\practice_pair\`
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\artifacts\caa\UNI_CAA01\behavior_L{10,16,22}.pt`
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\artifacts\caa\UNI_CAA01\condition_L{10,16,22}.pt`

**Missing directories (blockers):**
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\artifacts\caa\UNI_CAA01\calm_tails\`
- `c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\artifacts\caa\UNI_CAA01\pairs\`

**Bottom line:** Panic/divergence **source data exists** (3 tracer-ready invocations with full chaos). Natural pair **pipeline is built but empty** — calm tails, live miner, admitted pairs, and live neutral vectors are all still outstanding before ≥3 natural pairs + live neutral can be claimed.

[REDACTED]
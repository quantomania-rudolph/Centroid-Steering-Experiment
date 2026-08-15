# miner_gaps

Honest remaining gaps in the CAA proof pipeline / natural miner implementation.
Updated **2026-08-10** after thermonuclear review fixes, neutral live capture, and miner dry-run.

---

## Closed this session

| Gap | Status | Evidence |
|-----|--------|----------|
| **Phase 3B live miner body** | **Implemented** | `natural_miner.py` — scan → validate → capture → admit; `caa_mine_natural_pairs.py --scan/--dry-run/--live` |
| **Live neutral capture** | **Done** | `artifacts/gates/caa_neutral_purity_gate.json` `pass: true` (N1–N10 + N-S1/S2); `neutral_L{10,16,22}.pt` on XPU |
| **N5 neutral comparative geometry** | **Implemented** | `N5_panic_delta_alignment` — delta=cos(bad)-cos(calm) < 0.15 at L10/16/22 |
| **d-coherence (tracer vs extract)** | **Implemented** | `miner_pairing.check_d_coherence` in `natural_miner.mine_corpus` |
| **Calm-tail registry** | **Implemented** | `calm_tail_registry.py` + curated file for scheduler d=55 |
| **KV stamp on capture** | **Implemented** | `kv_stamp.kv_live_gate_pass` in capture path |
| **n_corpus_eligible in master gate** | **Populated** | From `trace_all_glob` in `caa_master_gate.py` |
| **assistant text from corpus** | **Fixed** | `resolve_assistant_text` decodes `chaos.token_ids` when session JSONL is preamble-only |
| **Bad T2 distress for chaos-only panic** | **Fixed** | `validate_caa_pair` accepts chaos distress evidence when lexical distress absent |

---

## Still open (blocks `production_ready=true`)

| Gap | Status | What it means |
|-----|--------|---------------|
| **≥3 admitted natural pairs** | **1 dry-run / 0 live** | Need live XPU capture + 2 more calm tails / invocations |
| **Live natural pair capture** | **In progress** | `CAA_MINE_LIVE=1 --live` on scheduler candidate; KV gate already pass |
| **Sparse legacy invocations** | **2 rejected** | `monitor_gate_stress`, `stem10_t08` — `token_ids_match=False`; need session JSONL token recovery or re-run |
| **More panic corpus** | **167 scanned, 3 eligible, 1 dense** | Need more Aider invocations with dense `chaos.token_ids` for mining scale |

---

## Trust-boundary / proof quality (deferred)

| Gap | Status | Notes |
|-----|--------|-------|
| **N3–N4 live replay** | Deferred | No per-task chaos replay at gen-start; prompt text scoring only |
| **KV10 cross-method numerical** | Weak | Structural check only |
| **rel_l2 tolerance** | Relaxed | `rel_l2_max=2e-3` vs plan `≤1e-4` |
| **Live vector identity on admitted pairs** | Fixture only | Not wired to pooled natural captures |
| **AGG6 re-verify at aggregate time** | Partial | Manual `aggregate_steering_vectors([caps])` bypass |
| **Cross-capture d / prompt_len consistency** | Not checked | Aggregate uses `captures[0]` |
| **NM-S3 sabotage** | Check CLI | Bad arm without stub+distress should fail T gates |
| **PR-S2 on practice path** | Indirect | Path guard only at write time |

---

## Downstream CAST (explicitly deferred)

| Gap | Status |
|-----|--------|
| **tau / sigma calibration** | Not started |
| **CAST live injection hook** | Not wired |
| **Production overwrite guard** | Partial |

---

## Data inventory (current)

| Asset | Count / path |
|-------|----------------|
| Neutral vectors | `artifacts/caa/UNI_CAA01/neutral_L{10,16,22}.pt` + `neutral_capture.json` |
| Neutral gate | `artifacts/gates/caa_neutral_purity_gate.json` — **pass** |
| Calm tails | `artifacts/caa/UNI_CAA01/calm_tails/live_panic_demo_scheduler__chatcmpl_b757de025c4c__d55.txt` |
| Natural pairs (live) | `pairs/natural/` — pending live capture |
| Corpus eligible | 167 scanned, 3 tracer-eligible, 1 dense candidate |
| Practice pair | `practice_pair/` — excluded from pool |

---

## Recommended close order

1. Complete live miner capture for scheduler d=55 → `pairs/natural/`.
2. Curate 2+ more calm tails (or re-run corpus with dense token_ids).
3. Admit ≥3 natural pairs with `kv_live_gate_pass=true`.
4. Wire live identity gate on pooled vectors.
5. CAST injection (separate track).

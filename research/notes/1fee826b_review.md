# Thermonuclear Review — Natural Pair Miner (Phase 3B), UNI_CAA01

## 0. Bottom line

`caa_mine_natural_pairs.py --live` is **not a miner**. It is a 34-line prerequisite checker (`run_live()`, lines 288–321) that reads two file-existence flags and writes a gate JSON saying "blocked." It never touches the tracer, never touches `calm_tail.py`, never calls activation capture, and never writes to `pairs/natural/`. Every other module it would need (`prefix_divergence_tracer.py`, `calm_tail.py`, `activation_capture.py`, `admission.py`, `capture_manifest.py`) is individually solid and unit-tested — the miner simply never calls them together. This is exactly what `research/notes/miner_gaps.md` already says, and my own tracing of the live corpus below confirms and sharpens it.

I ran the tracer against the actual corpus (`experiments/out/*_latest.json`, 167 invocations) — no files modified:

```
{"n": 167, "eligible": 3, "with_consensus": 3}
```

Only **1 of 3** eligible candidates has dense `token_ids` (`token_ids_match=true`): `divergence_corpus_stress_b_latest.json`, invocation `chatcmpl-b757de025c4c`, task `live_panic_demo_scheduler`, `d_consensus=55`, confidence `medium`. The other two (`monitor_gate_stress`, `stem10_t08_...`) are legacy-schema, sparse-token, `confidence` capped at `medium`, and their `prefix_text` decodes are garbage (words fused/missing — sparse token-id dict has gaps). This is the entire raw-ore inventory today: **one usable bad-arm candidate**, zero calm tails, zero natural pairs.

---

## 1. BUGS / BLOCKERS (file:line, severity)

### CRITICAL — miner body

1. **`research/caa_mine_natural_pairs.py:288-321` (`run_live`)** — Entire live path is a stub. No call to `trace_all_glob`/`trace_experiment`, no call to `calm_tail.build_calm_tail_from_file`, no call to `capture_matched_pair_activations`, no write under `pairs/natural/`. Returns `0` (success) even in the fully-blocked case — this is deliberate per plan, but means `run_caa_proof_gates.py:143` treats "blocked" as a clean pass forever, silently.
2. **`caa_mine_natural_pairs.py` — no import of `jlens_qwen.caa.calm_tail`** (confirmed via grep: only `research/caa_calm_tail.py` CLI imports it). The Phase-2 calm-tail builder exists and is unit-tested but is **completely disconnected** from the miner. There is no code path anywhere that goes "tracer bad candidate → look up curated calm file → `build_calm_tail_from_file` → `MethodBRecord(tail_kind='calm')`".
3. **No d-extraction / LCP wiring.** Plan invariant #6 (`caa_proof_pipeline_597a9ded.plan.md:116`): "Extraction `d` is token-LCP, not chaos-lag." `calm_tail.py:26-31` defines `longest_common_prefix()` but it is **only used inside `validate_prefix_lock`'s detail string** (`calm_tail.py:72`), never to actually compute `d_extract = longest_common_prefix(calm_ids, bad_ids)` for a real pair. No function in the codebase produces `d_extract` from two independent token streams for miner use.
4. **d-coherence check absent** (miner_gaps.md correctly flags this). Plan requires `|d_extract - tracer_d_consensus| ≤ 15` or admission fail (`plan.md:339`). Grep for `d_extract`/`tracer_d_consensus` in `caa_mine_natural_pairs.py` → zero matches. `validate_caa_pair.py`'s `_chaos_panic_near_d` (lines 39–51) is a *different* check (chaos-mass-lag vs text-lag within `validate_tail`), not the extraction-vs-tracer-consensus coherence check the plan and gaps doc both call out.
5. **KV-live stamping per capture never happens.** Grep for `kv_live_gate_pass =` assignment sites: only (a) test fixtures manually constructing `CaptureManifest(kv_live_gate_pass=True)` in `tests/unit/test_capture_manifest.py`, and (b) the fixture path in `caa_mine_natural_pairs.py:205` which hardcodes `False`. There is **no function** that reads `artifacts/gates/kv_equivalence_live_gate.json` (`pass: true`) and stamps a natural capture's `CaptureManifest.kv_live_gate_pass = True`. Even if `run_live` were implemented today, it has no mechanism to legitimately set this field to `True` for a real capture — `ADM3_kv_live` (`admission.py:114-117`) would fail closed forever.
6. **NM-S3 sabotage missing** (also flagged in miner_gaps.md). `run_fixture()` sabotage list (`caa_mine_natural_pairs.py:230-263`) has NM-S1, NM-S2, NM-S4 only. Plan (`plan.md:366`) requires NM-S3: "bad without stub+distress → bad T gates fail." Not implemented, not tested.
7. **`n_corpus_eligible` hardcoded `None`** — `research/caa_master_gate.py:113`. The tracer *can* produce this number today (`trace_all_glob()` → `eligible` count = 3), but master gate never calls the tracer; it's a dead field, always null, forever.

### HIGH — data/schema mismatches that will break a naive live implementation

8. **Corpus schema has no `divergence_hints` key.** I grepped all 4 `divergence_*_latest.json` files — zero occurrences of `"divergence_hints"`. `_estimators()` (`prefix_divergence_tracer.py:275-293`) and `eligible_for_panic_trace()` (lines 83-108) both fall back correctly to token/mass computation, so this *works today*, but any future miner code that assumes `chaos.divergence_hints.b_distress_rate` exists on real data (as the fixture synth in `caa_mine_natural_pairs.py:88` implies) will silently get `None`/`0.0` and mis-score. `validate_tail`'s T6/T9 checks (`validate_caa_pair.py:132-140, 154-162`) read `hints = chaos.get("divergence_hints") or {}` → always `{}` on real corpus → `b_d=0, b_s=0` → **T6/T9 always trivially pass on real bad-arm data regardless of actual chaos**, because the hint-based branch degrades to "0 < threshold" instead of computing from `tokens`. This is a **silent looseness bug**, not just a missing feature — on real corpus, the calm purity gates T6/T9 are not actually exercising the token-level chaos data at all if hints are absent and the caller doesn't separately verify tail rates. (T9 does compute `_tail_flag_rates` from tokens directly — that part is fine. But T6 for calm and the `hint_d/hint_s` branch of bad-arm T3 in `validate_caa_pair.py:208-219` are the ones vulnerable to a false-negative "hints present but empty" pattern if a future producer starts emitting an empty `divergence_hints: {}` dict rather than omitting the key — currently safe by luck of omission, not by design.)
9. **No `assistant` field on real invocations, only `assistant_preview` (truncated).** Confirmed via grep on `divergence_corpus_stress_b_latest.json`: no `"assistant":` key, only `"assistant_preview"` (visibly cut off mid-code-block at line 19). `resolve_assistant_text()` (`divergence_support.py:49-63`) correctly falls back to `assistant_from_session()` reading the paired `.jsonl` session file — and I confirmed the session file exists on disk (`experiments/out/aider_sessions/divergence_corpus_stress_b_20260809T193510Z.jsonl`). This path is *not broken*, but it is a **hidden hard dependency**: any live miner that reads only the `divergence_*_latest.json` and doesn't also pass `experiment_data={"session_path": ...}` into `trace_invocation`/`resolve_assistant_text` will silently score bad-arm text purity (`T1_bad_stub_score`, `T2_bad_distress_score`) against a truncated preview and can produce false negatives/positives. `caa_mine_natural_pairs.py` (not yet written) must explicitly thread `session_path` through.
10. **`prompt_len` naming mismatch.** Corpus invocation has `"prompt_tokens": 1948` (not `"prompt_len"`). `capture_manifest.py`, `admission.py`, `kv_equivalence.extract_positions` all expect `prompt_len`. A live miner must remap `inv["prompt_tokens"] → prompt_len` explicitly — no existing code does this mapping today; it's implicit tribal knowledge, not a function.
11. **No `calm_tails/` directory, no `pairs/natural/` directory exist on disk.** Confirmed via directory listing. `run_live`'s own blocker check (`caa_mine_natural_pairs.py:295-297`) correctly detects this and blocks — so the gate is *honest*, but this means **zero curation has happened**, matching miner_gaps.md's "Manual / undefined ops" status. There is no calm-tail curation for the one legitimate bad-arm candidate we have (`live_panic_demo_scheduler`, d=55).

### MEDIUM — cohesion / trust-boundary bugs (per miner_gaps.md, independently confirmed)

12. **`AGG6` re-verify bypass** — `steering_vectors.py:167-259` (`aggregate_steering_vectors`) only calls `assert_not_practice_path` on `capture_paths` if explicitly passed; `discover_natural_captures` (lines 105-143) does call `verify_on_disk`, but a caller who manually builds a captures list (as `caa_master_gate.py` would eventually need to for real aggregation) bypasses hash verification entirely. Confirmed: `aggregate_steering_vectors` itself never calls `verify_on_disk`.
13. **Cross-capture `d`/`prompt_len` consistency never checked.** `steering_vectors.py:190-191`: `d=captures[0].get("d")`, `prompt_len=captures[0].get("prompt_len")` — takes only the first capture's values with no equality assertion across the pool. With `n_natural_admitted ≥ 3` eventually, silently mixed `d` values would pool anyway.
14. **KV10 is structural, not numerical**, per its own docstring (`kv_equivalence.py:233-247` only checks presence of `divergent` key per arm/method, not an actual full-forward-vs-KV numeric comparison at the *bad* sequence's own full-forward — it's covered per-arm by KV8/KV9 already, so KV10 is redundant/weak as flagged).
15. **`rel_l2_max=2e-3`** (`kv_equivalence.py:129`) vs. plan's `≤1e-4` (`plan.md:96,267`) — confirmed relaxed, live proof passes at 7e-4 to 8.7e-4 (`kv_equivalence_live_gate.json`), i.e. within the *relaxed* budget but would fail the plan's original stated budget. This is a silently-lowered bar, not documented as a deviation anywhere except miner_gaps.md.
16. **PR-S2 (practice-path-mislabeled-as-natural) not actively tested at write time** — `assert_not_practice_path` exists and is called by `discover_natural_captures` (path-based, works), but there's no test that constructs a manifest with `source="natural"` physically under a `practice_pair/` directory and confirms `admission_checks`/`write_admission_gate` refuses to write it. `admission.py:124-127` (`ADM5_not_practice_path`) does check path string at admission time — this is fine defensively, but it's on the *admission* dict, not enforced inside `write_admission_gate`/`write_manifest` themselves, so a caller could skip `admission_checks()` and call `write_manifest()` directly on a practice path with `source="natural"` with zero resistance from `capture_manifest.py` itself (only `pool_eligible()` would later catch it downstream).
17. **Master gate `production_ready` doesn't require `caa_natural_pair_live_gate.json` pass explicitly** — `caa_master_gate.py:66-72` checks `n_natural >= 3` and several other gates but never checks `_passed("caa_natural_pair_live_gate.json")`. If `n_natural_admitted` ever reaches 3 via some out-of-band capture write (bypassing the live gate), `production_ready` could flip true without Phase 3B's own gate ever passing.

---

## 2. What a "true tracer miner" must do end-to-end

```
for each experiments/out/*_latest.json (or research/results/*):
  data = json.load(path)
  for inv in data["invocations"]:
    trace = trace_invocation(inv, experiment=data["experiment"], task_slug=data["task_slug"],
                              experiment_data=data, use_tokenizer=True)   # tokenizer REQUIRED live
    if not trace.eligible or trace.d_consensus is None:
        continue  # not a panic candidate
    if trace.confidence not in ("medium", "high"):
        continue  # or record as rejection reason
    d_tracer = trace.d_consensus
    prefix_ids = trace.prefix_token_ids           # from token_ids[:d] (dense) — REQUIRE dense token_ids
    if trace.token_ids_len is None or not trace.token_ids_match:
        reject("sparse/legacy schema — cannot admit")
        continue

    prompt_len = inv["prompt_tokens"]              # <-- explicit rename, not "prompt_len"
    bad_full_ids = trace's chaos["token_ids"]       # full bad completion
    bad_assistant = resolve_assistant_text(inv, experiment_data=data)  # session-file fallback

    calm_path = calm_tails_dir / f"{task_slug}__{invocation_id}__d{d_tracer}.py"  # or .txt
    if not calm_path.is_file():
        reject("no curated calm tail for this prefix"); continue

    calm_spec = build_calm_tail_from_file(
        prefix_token_ids=prefix_ids, d=d_tracer,
        calm_completion_path=calm_path, tokenizer=get_qwen_tokenizer(),
    )  # raises ValueError on prefix-lock failure -> reject, do not silently coerce

    # d-coherence (MISSING TODAY — must add)
    d_extract = longest_common_prefix(calm_spec.calm_full_token_ids, bad_full_ids)
    if abs(d_extract - d_tracer) > CHAOS_DEBOUNCE_STEPS:  # 15
        reject("d_extract/tracer_d_consensus mismatch"); continue

    spec = {
      "d": d_extract, "prefix_token_ids": calm_spec.prefix_token_ids[:d_extract],
      "calm_tail_token_ids": calm_spec.calm_tail_token_ids,
      "bad_tail_token_ids": bad_full_ids[d_extract:],
      "calm_tail_text": calm_spec.calm_text, "bad_tail_text": ...,
    }
    calm_run = {..., "chaos": <synthetic-or-forced-replay-chaos-for-calm>, "method_b_record": build_method_b_record(..., tail_kind="calm", prompt_len=prompt_len)}
    bad_run  = {..., "chaos": inv["chaos"], "method_b_record": build_method_b_record(..., tail_kind="observed", prompt_len=prompt_len)}

    pair = validate_pair(spec=spec, calm_run=calm_run, bad_run=bad_run, prompt_len=prompt_len)
    if not pair["pass"]:
        reject(pair["pair_checks"], pair["calm"]["checks"], pair["bad"]["checks"]); continue

    # Live capture — REQUIRES loaded ModelBundle, XPU, server stopped
    capture = capture_matched_pair_activations(
        bundle, prompt_ids, calm_spec.calm_full_token_ids, bad_full_ids, d_extract, layers=PROBE_LAYERS
    )
    manifest = CaptureManifest(source="natural", pair_slug=slug, d=d_extract, prompt_len=prompt_len,
                                calm_purity_pass=pair["calm"]["pass"], bad_purity_pass=pair["bad"]["pass"],
                                pair_gate_pass=pair["pass"],
                                kv_live_gate_pass=stamp_from_global_kv_live_gate())  # MISSING function today
    cap = verify_capture_integrity(capture, manifest=manifest)
    adm = admission_checks(pair_validate_pass=pair["pass"], cap_pass=cap["pass"], manifest=manifest,
                            capture_dir=NATURAL_ROOT / slug)
    if adm["pass"]:
        write everything under pairs/natural/<slug>/, write_manifest, write_admission_gate
        n_admitted += 1
    else:
        n_rejected += 1; rejection_reasons.append(...)

write_gate(caa_natural_pair_live_gate.json, pass_=n_admitted >= 1, extra={"n_candidates":..., "n_rejected":..., "rejection_reasons":[...]})
```

Missing pieces that must be **newly written** (not stitched from existing code): the `stamp_from_global_kv_live_gate()` helper, the `d_extract`/coherence check, the calm-tail file-naming/lookup convention, and forced-replay chaos generation for the calm arm (see gap #3 below — this is the "no live neutral capture"-adjacent hole: calm tails need their *own* chaos trace via teacher-forced replay through the live model to populate T6/T9/T11, not just token IDs).

---

## 3. Mining from the EXISTING panicked corpus without new Aider runs

**Partially possible, with a hard ceiling of 1 pair on current evidence.**

- The tracer already runs cleanly against existing JSON (I ran it — 3 eligible, 3 with consensus) — **no new Aider generation needed for the bad arm** of `live_panic_demo_scheduler` (`d=55`, dense `token_ids`, `confidence=medium`).
- The other 2 eligible candidates (`monitor_gate_stress`, `stem10_t08...`) are **legacy schema without dense `token_ids`** — `confidence` capped at `medium`, `prefix_text` decode is corrupted (word-fusion artifacts from sparse step-dict reconstruction). These should **not** be admitted as bad arms without re-running (or at minimum, re-deriving `prefix_token_ids` from the paired `.jsonl` session — the raw completion tokens might be recoverable from `messages` in the session JSONL even if the chaos JSON's per-step dict is sparse; worth checking `assistant_from_session` + re-tokenizing rather than assuming a full Aider re-run is required).
- What's still **missing and requires manual work, not new Aider runs**: a curated **calm tail** for the `d=55` prefix of `live_panic_demo_scheduler`. This is explicitly a human-authored artifact per plan invariant #5 — write a calm continuation of the Scheduler fix, encode with the Qwen tokenizer, confirm `encode(calm_text)[:55] == prefix_token_ids`. This is cheap (no XPU, no Aider) but is currently **zero-done** (`calm_tails/` doesn't exist).
- Live model access (XPU) **is** required for the KV capture step itself (`capture_matched_pair_activations`), even though the tracer/pairing/validation steps are pure CPU/JSON. So: mining from existing corpus gets you to a **validated matched-prefix spec with zero new Aider generations**, but the capture step still needs one XPU forward pass (cheap — one shared-prefill + two single-token decodes, not a full agentic run).
- **Realistic near-term ceiling: 1 admitted natural pair** from current corpus. `production_ready` needs ≥3 — that requires either recovering the two sparse-schema candidates' full token streams (from session JSONL, not a re-run) or genuinely new corpus runs.

---

## 4. Exact implementation plan to make the miner fully real

New/modified modules (none touched — plan only):

1. **`src/jlens_qwen/caa/miner_pairing.py`** (new) — pure functions, unit-testable without XPU:
   - `extract_d_from_lcp(calm_ids, bad_ids) -> int` — thin wrapper exposing `calm_tail.longest_common_prefix` for pairing use (currently private to validation detail strings).
   - `check_d_coherence(d_extract, tracer_d_consensus, *, debounce=15) -> CheckResult` — the missing NM/d-coherence gate.
   - `remap_invocation_fields(inv) -> dict` — normalize `prompt_tokens → prompt_len`, thread `experiment_data` for `resolve_assistant_text`.
2. **`src/jlens_qwen/caa/calm_tail_registry.py`** (new) — filesystem convention + lookup: `find_calm_tail(task_slug, invocation_id, d) -> Path | None` reading `artifacts/caa/UNI_CAA01/calm_tails/`; enforced naming so the miner doesn't have to guess.
3. **`src/jlens_qwen/caa/kv_stamp.py`** (new) — `read_global_kv_live_gate() -> bool` + `stamp_manifest_kv_live(manifest, *, gate_path=...)`. This closes gap #5 (KV stamping) cleanly and testably, separate from the miner's control flow.
4. **`research/caa_mine_natural_pairs.py`** — rewrite `run_live()`:
   - Iterate `trace_all_glob()` results (or a `--corpus-glob` CLI arg for targeting).
   - Filter `eligible and d_consensus is not None and token_ids_match` (reject sparse-schema candidates explicitly, with a rejection reason, not silently).
   - For each candidate: `find_calm_tail(...)`; if missing, append to `rejection_reasons`, continue (do not fail the whole run).
   - Build `spec`/`calm_run`/`bad_run` per §2 above; call existing `validate_pair`.
   - Add `check_d_coherence`.
   - Load `ModelBundle` (guarded by `CAA_MINE_LIVE=1` env flag, same pattern as `CAA_KV_LIVE=1`/`CAA_NEUTRAL_LIVE=1`), call `capture_matched_pair_activations`.
   - `stamp_manifest_kv_live` before `admission_checks`.
   - Write `pairs/natural/<slug>/` via existing `write_manifest`/`write_admission_gate` — no new I/O primitives needed, only orchestration.
   - Gate JSON must report real `n_candidates`, `n_rejected`, `rejection_reasons[]`, `n_admitted` (currently hardcoded 0/0 at lines 316-318).
5. **`caa_mine_natural_pairs.py` `run_fixture()`** — add NM-S3 sabotage case (bad-arm fixture without stub+distress flags → bad T gates must fail) alongside existing NM-S1/S2/S4.
6. **`research/caa_master_gate.py:113`** — replace hardcoded `n_corpus_eligible: None` with `len(trace_all_glob(use_tokenizer=False))`'s eligible count (cheap, no tokenizer needed for the count — pass `use_tokenizer=False` as I did in my own review run). Also add `_passed("caa_natural_pair_live_gate.json")` to the `production_ready` boolean (gap #17).
7. **`src/jlens_qwen/caa/steering_vectors.py`** — `aggregate_steering_vectors` should call `verify_on_disk` itself when `capture_paths` given (close AGG6 bypass), and add a cross-capture `d`/`prompt_len` agreement assertion before pooling (gap #13).
8. **Tests**: `tests/unit/test_miner_pairing.py` (new) for `check_d_coherence`, `extract_d_from_lcp`; extend `tests/unit/test_calm_tail_purity.py`-style sabotage coverage into a new `tests/unit/test_natural_pair_live_wiring.py` that mocks the ModelBundle/capture call so the orchestration logic is CI-testable without XPU (mirrors how KV structural vs live is split today).

---

## 5. Gaps needing more corpus vs. workable with existing data

**Workable with existing data (no new Aider runs):**
- Tracer eligibility/consensus computation — done, verified (3 eligible).
- Full pairing/validation logic for `live_panic_demo_scheduler` (d=55) — workable once a calm tail is manually authored.
- `d_extract`/LCP computation, d-coherence check — pure function work, no corpus needed.
- KV-stamp wiring — pure function work, no corpus needed (the live KV gate already passed).
- Fixing `n_corpus_eligible`, AGG6 bypass, cross-capture consistency, `production_ready` live-gate requirement — pure code fixes.

**Needs more corpus (or at least recovery work) — cannot be closed with current JSON alone:**
- Recovering usable bad-arm candidates for `monitor_gate_stress` and `stem10_t08_...` requires either re-deriving full token streams from their paired session `.jsonl` files (worth trying before assuming a re-run) or fresh Aider runs if that recovery fails.
- `production_ready` (`n_natural_admitted ≥ 3`) is structurally capped at 1 pair until either the above recovery succeeds or new panic-eliciting Aider sessions are generated — this is an evidence gap, not a code gap.
- Phase 4 N5 (neutral comparative geometry) is currently deferred to a "golden fixture" per plan — re-running N5 on real admitted natural captures needs at least the 1 pair above to exist first.
- Phase 6 identity live re-run similarly needs ≥1 real admitted pair before it can move past fixture-only.

**Everything else — the stub `run_live`, the missing d-coherence check, the disconnected calm-tail module, the unstamped KV-live field, the missing NM-S3 sabotage, the hardcoded `n_corpus_eligible` — is pure engineering debt with zero corpus dependency.** The corpus problem (1 usable candidate) and the code problem (miner is a stub) are separate blockers, and fixing the code problem will not by itself produce ≥3 admitted pairs.

I'm seeing that caa_mine_natural_pairs.py never imports the calm_tail module and kv_live_gate_pass is only set to True in unit test constructors, never in the actual production code path. This reveals a gap where real captures aren't being stamped with kv_live_gate_pass=True based on the global kv_equivalence_live_gate.json validation—it's all manual and never actually wired up.

I also don't see any "d_extract" or "d-coherence" checks anywhere in the miner. Now I have what I need to write the final report. I'm verifying that the assistant field resolution works through validate_pair→validate_tail→trace_invocation, so for the live path the bad_run's invocation needs the full assistant text, which is resolvable from the session jsonl since it exists there. The completion_tokens fields check out too—trace_invocation expects invocations with "completion_tokens", "chaos", and "id"/"invocation_id" keys, and the real corpus invocations already have exactly these fields, so trace_experiment works directly on them. I confirmed this when we ran the tracer against the real corpus and got eligible=3. Now I'm checking the assistant_preview truncation issue—that field appears to be cut off. Looking at whether the miner actually computes the token-level longest common prefix between calm and bad token streams to derive the extraction depth, or if it's just using a predetermined value — the helper function exists in calm_tail.py but doesn't seem to be called anywhere in the validation or mining logic to actually calculate this derived d_extract from real token pairs.
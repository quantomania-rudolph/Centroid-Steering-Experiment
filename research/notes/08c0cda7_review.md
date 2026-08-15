# Thermonuclear Review — UNI_CAA01 §2.4 Neutral Extraction

Scope read: `neutral_capture.py`, `caa_extract_neutral.py`, `test_neutral_capture.py`, `steering_vectors.py` (`remove_neutral_component`/`aggregate_steering_vectors`), `activation_capture.py`, `capture_manifest.py`, `caa_extract_steering_vectors.py`, `true_UNICAA01_cast.md` §2.4, `miner_gaps.md`, and the CAA proof pipeline plan (Phase 4). Also traced provenance of the actual `NEUTRAL_TASKS` back to `experiments/out/aider_chaos_proof_meta_wipe_*.json`.

---

## 1. BUGS (file:line, severity)

### CRITICAL

**C1 — `chaos_proof_meta_wipe` is not a neutral task; it's a rebranded distress-pressure task.**
`src/jlens_qwen/caa/neutral_capture.py:24-31`. Half of the entire "neutral" reference set is this task. Its real provenance (`experiments/out/aider_chaos_proof_meta_wipe_latest.json:6`) is:

> `"The pylint tool is broken on this machine. Fix chaos_proof_meta_wipe/main.py so that is_palindrome works. Keep lint-clean. Never write shell commands or pip install into the file."`

and the source docstring for that module is literally *`"""Palindrome helper — incomplete on purpose for distress-lexicon pressure."""`*. This slug was purpose-built to bait apology/pip/broken-environment language — exactly the class of task the plan explicitly bans for `research_topo_broken` ("**Do not use** ... stress/trajectory — not neutral"). Someone rewrote the `user_prompt` in `NEUTRAL_TASKS` to a sanitized string ("Ensure is_palindrome('racecar') returns True. Lint-clean. Never pip.") so it passes the N1 text scan, but reused the *same slug/task identity* whose entire reason for existing was distress-lexicon pressure. N2 ("`N2_task_set`" in `caa_extract_neutral.py:38-45`) only checks the slug string is in `{"chaos_proof_clean","chaos_proof_meta_wipe"}` — it has zero ability to detect that the second slug is contaminated by design. This is exactly the sabotage class N-S1/N2 are supposed to catch, and it slips through because the check operates on the sanitized rewritten text, not on the task's actual history/identity.

**C2 — N3–N6 are hardcoded `pass: True` placeholders, in BOTH unit and live modes — false-green gate.**
`research/caa_extract_neutral.py:67-75` and `:116`. `main()`'s live branch (`CAA_NEUTRAL_LIVE=1`) computes `result = capture_neutral_gen_start(bundle)` but then calls the *same* `_unit_gate()` used for the no-model unit path (line 116: `payload = _unit_gate()`), which never inspects `result.neutral`, never computes `cos(h_task_i, û_neutral)` (N4), never compares vs golden calm/bad (N5), never computes `cos(û_neutral, û_cond)` (N6). The `detail` string claims `"covered in test_neutral_capture / live when CAA_NEUTRAL_LIVE=1"` — this is **false**; grep of `test_neutral_capture.py` confirms N4/N6/N7 are only exercised against synthetic orthonormal basis vectors (`test_n4_n6_n7_synthetic_geometry`), never against the real `capture_neutral_gen_start` output. The persisted gate at `artifacts/gates/caa_neutral_purity_gate.json` shows exactly this: `"mode": "unit"`, `pass: true`, with 4 of 8 checks being decorative always-true entries.

**C3 — N-S2 sabotage claim is hardcoded, not exercised.**
`research/caa_extract_neutral.py:61-65`. `{"id": "N-S2", "pass": True, "detail": "single-task rejected by capture_neutral_gen_start"}` is asserted without ever calling `capture_neutral_gen_start` with one task inside `_unit_gate()`. `write_gate()` treats sabotage as mandatory proof-of-catch (`gate_io.py:37-45`), but a hardcoded `True` satisfies that mandatory-sabotage requirement without any actual sabotage being run in this invocation. The real test exists only in `tests/unit/test_neutral_capture.py:39-41` (a separate execution context) — the gate JSON itself is not evidence of anything.

**C4 — No per-layer sample-count guarantee; `neutral[L]` can silently degrade to n=1.**
`src/jlens_qwen/caa/neutral_capture.py:118-129`. The only multiplicity guard is `if len(per_task) < 2: raise` (task-level). Nothing checks `len(accum[L]) >= 2` per layer before `torch.stack(accum[L]).mean(...)`. If one task's capture is missing layer L (e.g., `captured.get(L, {}).get(gen_pos)` returns `None` at `:111`), that layer's mean silently comes from a single task while `NeutralCaptureResult.n_tasks` still reports 2 — directly violating the N-S2 invariant ("û_neutral requires ≥2 neutral tasks") at the level that actually matters (per-layer geometry), while every high-level check still reports pass.

**C5 — No round-trip/tokenization verification of the captured position (no T10-equivalent for neutral).**
`src/jlens_qwen/caa/neutral_capture.py:91-99`. `start_ids = tokenizer.encode(task["assistant_start"], add_special_tokens=False)` tokenizes the continuation text **standalone**, then splices `start_ids[0]` onto the end of the chat-templated prompt (`full_ids = prompt_ids + [start_ids[0]]`). There is no check that this spliced token is the same token the tokenizer would produce when encoding the *actual* concatenated string (prompt text + assistant preamble + continuation), which is exactly the failure mode the plan's calm-tail work explicitly guards against with a "round-trip lock" (`tokenizer.encode(calm_text)[:d] == prefix_token_ids`, Phase 2). BPE merge boundaries around a leading `` ``` `` after a newline are a classic case where isolated-string tokenization and in-context tokenization diverge. If they diverge here, `gen_pos` captures the hidden state associated with a token that was never actually validated to be the true first assistant token — contaminating `û_neutral` with an unverified/possibly-wrong position, and no test or gate anywhere would catch it (all tests are tokenizer-free / synthetic).

**C6 — N7's "unit-norm" and "exactly these 3 layers" claims are never checked against the actual output.**
`research/caa_extract_neutral.py:47-51`: `N7_layers` checks `list(PROBE_LAYERS) == [10, 16, 22]` — a tautological check on a module-level constant that can never fail regardless of what `capture_neutral_gen_start` actually produced. It does **not** check `set(result.neutral.keys()) == {10,16,22}` (which, per C4/`neutral_capture.py:124-125`, could be a strict subset if `accum[L]` was empty for some layer) and does **not** check `abs(norm(vec) - 1) < eps` for any produced vector anywhere in the codebase (unit-norm is never asserted post-hoc, only computed via a division that itself has an unguarded fallback — see C7).

### HIGH

**H1 — Silent non-unit-norm fallback breaks N7 exactly when it matters most.**
`src/jlens_qwen/caa/neutral_capture.py:128-129`:
```python
n = float(mean_h.norm().item())
neutral[L] = (mean_h / n).tolist() if n > 1e-12 else mean_h.tolist()
```
If the two neutral tasks' mean hidden state nearly cancels (small norm — plausible if the two tasks pull in different directions and only partially agree, which is likely given C1's contamination), the function returns the **raw, non-unit-norm** vector rather than raising. This is precisely the degenerate case (tiny, meaningless mean direction) that should fail closed, and instead it silently ships a broken vector that violates N7 without any downstream check catching it (per C6).

**H2 — `N2` in `_unit_gate` is not the plan's N2.** `research/caa_extract_neutral.py:38-45` vs. plan Phase 4 table (`N2 | Each task forced replay: eligible_for_panic_trace==false, no B_STUB_DISTRESS`). The code's "`N2_task_set`" checks only that the slug set matches two fixed strings — an entirely different, much weaker check squatting on the ID `N2`. The actual N2 (chaos-replay purity) is never implemented anywhere in this module. Anyone reading `"N2_task_set": pass=true` in the gate JSON will incorrectly believe the plan's N2 requirement is satisfied.

**H3 — Storage schema mismatch between the two `neutral_L*.pt` writers.**
`research/caa_extract_neutral.py:102-112` writes `{"vector_id","layer","kind":"neutral","u_hat","n_tasks"}`. `src/jlens_qwen/caa/steering_vectors.py:327-339` (`save_steering_artifacts`, invoked from `caa_extract_steering_vectors.py:137-142` production path) writes to the **same directory** (`DEFAULT_OUT == NEUTRAL_DIR`) with `{"vector_id","layer","kind":"neutral","u_hat","n_pairs"}` — different key (`n_tasks` vs `n_pairs`), and it **overwrites** the file produced by the neutral-capture step. No version tag, no schema check, no protection against the aggregate-extraction step silently clobbering the neutral-capture artifact with re-derived metadata under a different field name. `_normalize_capture_format` (`steering_vectors.py:380-431`) and `_load_neutral()` (`caa_extract_steering_vectors.py:29-40`) tolerate this today only because they happen to only read `u_hat`, but this is fragile-by-luck, not by design.

**H4 — `test_n4_n6_n7_synthetic_geometry` is a placebo test for N4/N6/N7.**
`tests/unit/test_neutral_capture.py:44-57`. It builds hand-crafted orthonormal basis vectors (`e0,e1,e2`) and checks `cos` between them is `<0.50` — true by construction, independent of any code under test — and checks `remove_neutral_component` removes an *exact*, deterministic projection (also true by construction/algebra, not by any learned property). It never calls `capture_neutral_gen_start`, never touches `NEUTRAL_TASKS`, never touches a real model. The test name implies N4/N6/N7 coverage; the reality is it tests that `torch.dot` and subtraction work.

**H5 — `test_neutral_purity_gate` is a placebo test for the gate itself.**
`tests/unit/test_neutral_capture.py:65-77`. It builds a fully synthetic checks/sabotage list (all `pass: True`, `N1..N7` labels invented inline) and asserts `write_gate` returns `pass=true`. This tests `write_gate`'s generic aggregation logic, not `caa_extract_neutral.py`'s actual `_unit_gate()` construction or `capture_neutral_gen_start`'s output. No test in this file ever invokes `_unit_gate()` from `caa_extract_neutral.py` to check its real (buggy, per C2/H2) content.

### MEDIUM

**M1 — Two neutral tasks are drawn from near-identical short-kata coding domains.** `chaos_proof_clean` (fibonacci) and `chaos_proof_meta_wipe` (palindrome) are both trivial single-function algorithmic edits with nearly identical prompt templates ("Edit X/main.py only... Lint-clean"). N=2 samples from one narrow style is an extremely thin basis for "the neutral coding direction"; N4 (mean cosine coherence ≥0.60 across tasks) is exactly the check that would surface this thinness — and it's never run (C2), so this risk is currently invisible.

**M2 — `neutral_raw` is stored but never audited.** `neutral_capture.py:41,127` stores the un-normalized mean but nothing ever inspects its norm to detect the H1 degenerate-cancellation case before normalization, nor logs a warning when it's small.

**M3 — `encode_chat_prompt` fallback path (`neutral_capture.py:55-68`) for tokenizers without `apply_chat_template` uses a hand-rolled template (`"User: {p}\nAssistant:"`) that has no relationship to the real Qwen chat template used everywhere else in the pipeline (`<|im_start|>...`). If this fallback is ever silently taken (e.g., a mocked/broken tokenizer in some test harness), `gen_pos` will be computed against a completely different prompt structure with no warning.

**M4 — `capture_neutral_gen_start` accepts `layers` from the caller with no validation against `PROBE_LAYERS`.** `neutral_capture.py:76,86`. Nothing stops a caller from passing `layers=[5]` and getting a "neutral" result whose keys silently violate N7's `[10,16,22]` requirement — again unguarded because N7 checks the global constant, not the actual returned dict (C6).

---

## 2. Stubs that must become real

| Stub | Where | Must become |
|---|---|---|
| N3 "unit_deferred_or_fixture" | `caa_extract_neutral.py:68-75` | Real per-task chaos replay: `eligible_for_panic_trace==false`, `b_distress_rate==0`, `b_stub_rate==0` on the *full generated stream*, not just the prompt text |
| N4 "unit_deferred_or_fixture" | same | Real `mean(cos(h_task_i, û_neutral)) ≥ 0.60` computed from `result.per_task[*]["h_L{L}"]` against `result.neutral[L]`, per layer, in the live gate |
| N5 "unit_deferred_or_fixture" | same | Build the golden calm/bad fixture referenced by the plan (does not exist anywhere in the repo today — confirmed by search) and compute `cos(û_neutral, μ_calm) > cos(û_neutral, μ_bad) + 0.10` |
| N6 "unit_deferred_or_fixture" | same | Real `cos(û_neutral, û_cond) < 0.50`, wired to run automatically whenever `condition_L*.pt` exists (currently no such wiring exists at all — not even manual) |
| N-S2 hardcoded sabotage | `caa_extract_neutral.py:62-65` | Actually invoke `capture_neutral_gen_start(bundle_or_stub, tasks=[NEUTRAL_TASKS[0]])` inside `_unit_gate()`/live gate and capture the real raised/not-raised outcome |
| "N2_task_set" mislabeled check | `caa_extract_neutral.py:38-45` | Rename to something like `N2_task_slugs_allowlist` (it is not N2) and implement the real N2 (forced-replay chaos purity) separately with its own ID |
| `test_n4_n6_n7_synthetic_geometry` | `test_neutral_capture.py:44-57` | Replace/extend with a test that runs `capture_neutral_gen_start` against a fixture/mock bundle producing deterministic fake hidden states per task, then asserts N4/N6/N7 on the *actual return value* |
| `test_neutral_purity_gate` | `test_neutral_capture.py:65-77` | Must call the real `_unit_gate()` from `caa_extract_neutral.py` (import it) rather than fabricating an independent checks list |
| Round-trip tokenization check | missing entirely | New check (proposed `N8` below) verifying `full_ids` reproduces true in-context tokenization |
| Per-layer sample-count guard | missing entirely | New check (proposed `N9` below) |

---

## 3. Exact proof checks needed for "indisputable neutral"

In addition to fixing N1/N2/N7 as specified above, add:

- **N8 — Tokenization integrity.** For each task, assert `tokenizer.encode(prompt_text_rendered + assistant_start, add_special_tokens=False)[:len(prompt_ids)+1] == full_ids` (or equivalent in-context re-tokenization check), proving the spliced `gen_pos` token is the token the tokenizer would actually produce in context — not an isolated-string artifact.
- **N9 — Per-layer multiplicity.** Assert `len(accum[L]) == len(per_task) >= 2` for every `L` in `layers` before computing any mean; raise (don't silently skip) if any layer is short.
- **N10 — Unit-norm postcondition.** Assert `abs(||neutral[L]|| - 1.0) < 1e-5` for every `L` in the *actual* output dict, and assert `set(neutral.keys()) == set(layers)` (catches C4/C6/H1 in one shot).
- **N11 — Task provenance/history check.** Before admitting any slug into `NEUTRAL_TASKS`, require a recorded provenance note proving the task was never designed for stress/distress pressure — this must be a human-reviewed, git-tracked assertion, not a runtime text scan, since C1 shows the runtime text scan (N1) is blind to task *intent* when the surface text has been sanitized. Practically: replace `chaos_proof_meta_wipe` with a freshly authored, provenance-clean palindrome/topo/whatever task that has no prior history as a pressure probe, and record that provenance in a comment + this notes file.
- **N4/N5/N6 must run against real tensors** (see §2) — not synthetic e0/e1/e2 vectors and not hardcoded pass=True.
- **Gate honesty check** — any check whose `detail` claims something is "covered elsewhere" must be either (a) removed from the checks list entirely, or (b) actually re-executed at gate-write time. `write_gate`'s mandatory-sabotage rule should also be tightened so a sabotage check can only be marked `pass=True` if it was computed in the same function call, not asserted.

---

## 4. Recommended fix list (priority order)

1. **Replace `chaos_proof_meta_wipe`** with a genuinely provenance-clean second neutral task (C1). This is the single highest-priority fix — it currently contaminates 50% of the neutral basis with a task designed to bait distress language.
2. **Kill the false-green N3–N6 placeholders** in `caa_extract_neutral.py`; make the live path actually compute them against `result` (C2), and stop reusing `_unit_gate()` unmodified for the live gate.
3. **Fix N-S2's fake sabotage** — actually execute the single-task call inside the gate function (C3).
4. **Add N9 (per-layer multiplicity guard)** and **N10 (unit-norm + exact-layer-set postcondition)** to `capture_neutral_gen_start`/the gate (C4, C6, H1).
5. **Add N8 (tokenization round-trip check)** to `capture_neutral_gen_start` before trusting `gen_pos` (C5).
6. **Fix H1's silent non-unit fallback** — raise instead of returning a non-unit vector when `mean_h.norm() <= 1e-12`.
7. **Rename/repair the mislabeled `N2_task_set`** and implement the real N2 (chaos-replay purity) as its own check (H2).
8. **Unify the `neutral_L*.pt` schema** between `caa_extract_neutral.py` and `steering_vectors.save_steering_artifacts` (same key, e.g. always `n_pairs`, with a `source: "neutral_capture"|"aggregate_resave"` tag) so the two writers can't silently disagree (H3).
9. **Replace the placebo tests** (`test_n4_n6_n7_synthetic_geometry`, `test_neutral_purity_gate`) with tests that exercise `capture_neutral_gen_start` and `_unit_gate()` for real, using a fixture/mock `ModelBundle` (H4, H5).
10. Build the missing **golden calm/bad fixture** required by N5 (does not exist anywhere in the repo today) so N5 can be implemented at all.
11. Only after 1–10: run live neutral capture on XPU (`CAA_NEUTRAL_LIVE=1`) — running it today would produce a `neutral_L*.pt` that is unverifiable by every metric that matters (N4/N5/N6/N8/N9/N10 all unenforced) and half-derived from a distress-pressure task.

No files were modified as part of this review.
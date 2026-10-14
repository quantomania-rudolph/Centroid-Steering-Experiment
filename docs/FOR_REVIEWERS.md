# For reviewers (start here)

**Project:** [Centroid Steering Experiment](../README.md) — portfolio work demonstrating **activation steering** for a local coding agent (skills demo, not a product or randomized benchmark).

**Goal of this page:** Map **resume / README claims** → **files and commands** so you can verify in minutes without reading the whole repo.

**Frozen numbers:** [resume_metrics_snapshot.json](resume_metrics_snapshot.json) (regenerate: `python research/compute_resume_stats.py` from repo root with `.venv` active).

---

## Five-minute check (no GPU)

From repo root, with venv activated and dependencies installed:

```powershell
python research/compute_resume_stats.py
```

You should see `live_cast_pairs 15`, `demo_panic 12 15`, `demo_neutral 5 15`, and `wrote docs/resume_metrics_snapshot.json`.

**Clone note:** `experiments/unicaa01_corpus/` session JSON is **gitignored** (large/local). On GitHub, trust the committed [resume_metrics_snapshot.json](resume_metrics_snapshot.json) and gate JSON under `artifacts/gates/`; re-run `compute_resume_stats.py` only on a machine that has corpus outputs. CI runs [validate_resume_snapshot.py](../research/validate_resume_snapshot.py) + unit tests instead.

Then open:

1. [resume_metrics_snapshot.json](resume_metrics_snapshot.json) — headline metrics + **per-pair** `demo_panic_pass` / `demo_neutral_pass`
2. [artifacts/caa/UNI_CAA01/metadata.json](../artifacts/caa/UNI_CAA01/metadata.json) — `"n_pairs": 39`, `"alpha_0": 0.85`
3. [artifacts/gates/caa_alpha0_4c_gate.json](../artifacts/gates/caa_alpha0_4c_gate.json) — top-level `"pass": true`
4. [artifacts/gates/cast_oneoff_panic_live_proof.json](../artifacts/gates/cast_oneoff_panic_live_proof.json) — `"pass": true`
5. [artifacts/gates/cast_oneoff_b02_live_proof.json](../artifacts/gates/cast_oneoff_b02_live_proof.json) — `"pass": true`

**Scope honesty:** [artifacts/gates/caa_steering_status.json](../artifacts/gates/caa_steering_status.json) has `"production_ready": false` — intentional; this is a research/portfolio harness.

---

## Headline claim (matches resume)

> On a **15-task** slice with **verified live steering** in logs, **80%** vs **33%** pass small automated `demo()` checks vs neutral twins on the same tasks (**+46.7 pp**, **~2.4×**).

| Check | Definition |
|--------|------------|
| **Which 15 tasks?** | Listed in snapshot as `live_cast_pair_ids` and `pair_details` |
| **“Verified live steering”** | Panic corpus session: `cast_enabled`, `cast_alpha_0 = 0.85`, ≥1 `inject_applied` token (see `headline_inclusion_rule` in snapshot) |
| **Demo pass** | Script runs `demo()` on on-disk `qwen code/<slug>/main.py` after full Aider session; expected values in `research/compute_resume_stats.py` → `ORACLES` |
| **Neutral twin** | `experiments/unicaa01_corpus/out/unicaa01_corpus_<pair>_neutral_latest.json` for same `pair_id` |
| **Not claimed** | Full 66-pair corpus, inv1-only scoring, faster wall-clock inference, production deployment |

---

## Resume bullets → evidence

Use the same wording as [CENTROID_STEERING_EXPERIMENT_RESUME.md](CENTROID_STEERING_EXPERIMENT_RESUME.md).

| # | Resume claim (short) | Verify here |
|---|----------------------|-------------|
| **1** | End-to-end steering pipeline; **39** prefix-aligned pairs; layer-16 inject; session logging | `metadata.json` (`n_pairs`, vectors under `artifacts/caa/UNI_CAA01/`); server hook: `experiments/serve_qwen_openai.py`; sessions: `experiments/unicaa01_corpus/out/*_latest.json` |
| **2** | **Conditional** steering; inject on panic pattern, quiet on clean controls | `artifacts/gates/cast_stream_live_gate.json`, `cast_inject_live_gate.json` (`"pass": true`); logic: `src/jlens_qwen/caa/cast_stream_gate.py`, `cast_injector.py` |
| **3** | Calibrated **α₀ = 0.85**; step-down / holdout harness; preflight | `metadata.json` `alpha_0`; `artifacts/gates/caa_alpha0_4c_gate.json`; `research/prove_cast_alpha0_4c_live.py`, `src/jlens_qwen/caa/cast_run_preflight.py` |
| **4** | **129** instrumented corpus sessions; **80% vs 33%** on **15** inject-verified pairs | `artifacts/caa/UNI_CAA01_corpus/scan_report.json` (`n_files` 129); metrics: `python research/compute_resume_stats.py` |
| **5** | **Two** out-of-corpus tasks (min-heap, min-stack) | `research/prove_cast_oneoff_live.py`; gates `cast_oneoff_panic_live_proof.json`, `cast_oneoff_b02_live_proof.json`; code under `qwen code/cast_oneoff_panic/`, `cast_oneoff_b02/` |
| **6** | Unit tests for corpus policy, CAST preflight, Aider watchdog | `tests/unit/test_unicaa01_corpus_policy.py`, `test_cast_run_preflight.py`, `test_aider_watchdog.py`, `test_cast_oneoff_task.py`, `test_alpha0_4b_harness.py` — run: `pytest tests/unit/test_unicaa01_corpus_policy.py tests/unit/test_cast_run_preflight.py tests/unit/test_aider_watchdog.py -q` |

---

## Optional: full live re-run (GPU / XPU + time)

Requires model weights, Intel Arc XPU stack, and a running server — see [experiments/README.md](../experiments/README.md).

```powershell
python research/prove_cast_oneoff_live.py --task panic
```

This rewrites `artifacts/gates/cast_oneoff_panic_live_proof.json` if the harness passes.

---

## Architecture (30 seconds)

```text
Broken stub → Aider → Qwen server (per-token monitors)
              → if gated: add contrast vector at layer 16
              → session JSON + demo oracle + gate JSON
```

Narrative + caveats: [PROJECT_BRIEF.md](PROJECT_BRIEF.md).  
Copy-paste bullets + sound bite: [CENTROID_STEERING_EXPERIMENT_RESUME.md](CENTROID_STEERING_EXPERIMENT_RESUME.md).

---

## What we improved for reviewers (meta)

| Pain point | Fix in repo |
|------------|-------------|
| Claims scattered | This page + README claim table |
| “Where did 80% come from?” | `compute_resume_stats.py` + per-pair rows in snapshot |
| Gate file overload | README lists **five** primary gates; full set under `artifacts/gates/` |
| Resume ≠ GitHub | Same numbers and bullet IDs in README, resume doc, and snapshot |
| Setup buried | README keeps install **below** verification |

---

## Security

Do not commit personal access tokens. If a token was ever pasted in chat, revoke it in GitHub → Settings → Developer settings.

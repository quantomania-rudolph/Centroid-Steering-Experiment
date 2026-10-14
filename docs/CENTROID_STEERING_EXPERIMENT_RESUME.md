# Centroid Steering Experiment

**What this is:** A portfolio project that shows end-to-end work on **activation steering** for a coding LLM—data, vectors, when to intervene, live agent runs, and honest measurement. It is a **skills demonstration**, not a shipped product or a peer-reviewed benchmark.

**Model:** Qwen2.5-Coder-7B-Instruct (local on Intel Arc XPU)  
**Agent loop:** Aider + custom OpenAI-compatible server with per-token logging  
**Last updated:** 2026-10-14 · **Live metrics:** [resume_metrics_snapshot.json](resume_metrics_snapshot.json) (from `python research/compute_resume_stats.py`)

**Names:** Public title **Centroid Steering Experiment** (see root [README.md](../README.md)). GitHub may still list the repo as `J-Do-It` until renamed; local folder may be `J-Do It`. Python package: `jlens-qwen`.

---

## Headline result (use on resume)

On **15 matched coding tasks** where steering actually fired in the logs:

| | With conditional steering | Same tasks, no steering |
|---|--:|--:|
| Small `demo()` checks pass | **12 / 15 (80.0%)** | **5 / 15 (33.3%)** |
| Lift | **+46.7 pp** | — |
| Relative success | **2.4×** | baseline |

**Plain English:** Same broken Python puzzles, same model, same agent. One run only nudged the model when internal signals looked like “stuck on a stub + stressed.” That side produced working fixes **2.4×** as often on this slice (verified 2026-10-14).

**One line for LinkedIn / resume:**

> Built conditional activation steering for a local coding agent; on a 15-task evaluation slice, demo success rose from **33% to 80%** vs non-steered twins (repro: `python research/compute_resume_stats.py`).

**Be honest in interviews:** This is a **subset** with verified inject in session logs—not all corpus tasks, not a randomized study. Scores use automated `demo()` oracles on final on-disk code after full Aider sessions.

---

## What I’m demonstrating (skills)

| Skill area | What the repo shows |
|------------|---------------------|
| **Steering vectors** | Contrast (“panic” vs “calm”) directions in activation space; layer-16 behavior vector from matched agent traces |
| **When to steer** | Gated injection—nudge only when monitors agree it’s the right moment, not on every token |
| **Agent + inference** | Live decode hook, session JSON, Aider integration, long-run stability on constrained hardware |
| **Rigor** | Automated pass/fail gate JSON, sabotage checks, one-off tasks outside the training set |
| **Communication** | This doc + reproducible scripts instead of hand-waved plots |

---

## How it works (short)

```text
Broken Python task → Aider asks the model to fix one file
        ↓
While tokens stream out: lightweight monitors score “stub-like” vs “stressed” text
        ↓
If rules say “intervene now” → add a precomputed steering direction in the network
        ↓
Log everything; run small demo tests; write gate artifacts
```

**Steering idea in one sentence:** Train a direction that separates bad vs good coding traces, then add that direction **only** when live signals match the condition you trained on—similar in spirit to contrastive activation steering (CAA) with conditional gating (CAST-style).

**Prefix alignment (why it matters):** Vectors are only meaningful if you compare activations at the **same point** in generation. The pipeline finds where panic traces diverge from calm ones (often around **~8 tokens** in this corpus) and builds pairs from shared prefixes.

---

## Numbers worth citing (with caveats)

Recompute: `python research/compute_resume_stats.py`

| Metric | Steered (panic arm) | Neutral twin | Notes |
|--------|--------------------:|-------------:|-------|
| Demo pass (15 pairs) | 80.0% | 33.3% | Matched tasks, verified inject |
| Demo lift | +46.7 pp | — | Panic − neutral on same pair IDs |
| Avg completion tokens (inv1 in logs) | 116 | 330 | **64.9%** fewer tokens on steered arm (slice) |
| Corpus sessions scanned | 129 (56 panic / 60 neutral OK) | — | `scan_report.json` |
| Natural contrast pairs | 39 | — | `metadata.json`, α₀ = **0.85** |

**Gates (in-repo, all pass):** stream live, inject live, alpha0 4c, one-off panic, one-off b02 — see `gates_pass` in [resume_metrics_snapshot.json](resume_metrics_snapshot.json).

**Out-of-distribution:** Min-heap (`cast_oneoff_panic_live_proof.json`) and min-stack (`cast_oneoff_b02_live_proof.json`) — inject + demo pass (2026-08-18 harness runs).

**Do not claim:** 100% corpus success, faster inference, or production copilot deployment (`production_ready` remains false in pipeline metadata).

---

## Resume bullets (copy-paste)

1. Built an **end-to-end activation steering** pipeline for Qwen2.5-Coder-7B on Intel Arc: contrast vectors from **39** prefix-aligned agent traces, live inject at layer 16, full session logging.

2. Implemented **conditional steering** so injection runs when stream monitors match trained “stub + distress” patterns—not on clean control runs (automated gate proofs in `artifacts/gates/`).

3. Calibrated injection strength (**α₀ = 0.85**) with step-down and holdout checks; fail-closed preflight before long agent jobs.

4. Ran **live Aider experiments** over **129** instrumented sessions; on a **15-task** matched slice with verified inject, **80%** demo pass vs **33%** without steering.

5. Validated on **two unseen** coding tasks via `research/prove_cast_oneoff_live.py` (gates pass, demos pass).

6. Added **unit tests** and gate runners for corpus policy, preflight, and agent stall watchdogs on long XPU generations.

---

## Sound bite (~30 seconds)

“I built a local coding-agent setup that can **steer** the model mid-generation: first I learned a direction from calm vs panic traces, then I only applied it when live monitors said the model was in the right failure mode. I measured it with small demo tests and automated gates—not as a paper, but to show I can take steering from idea to logged, reproducible runs. On one honest 15-task slice, steered runs beat non-steered twins 80% to 33%.”

---

## Repro pointers (for GitHub / reviewers)

| Goal | Command or path |
|------|-----------------|
| Recompute headline stats + snapshot JSON | `python research/compute_resume_stats.py` |
| Frozen metrics copy | `docs/resume_metrics_snapshot.json` |
| One-off live proof | `python research/prove_cast_oneoff_live.py --task panic` |
| Vector + α₀ metadata | `artifacts/caa/UNI_CAA01/metadata.json` |
| Calibration gate (example) | `artifacts/gates/caa_alpha0_4c_gate.json` |
| Deeper engineering brief | `docs/PROJECT_BRIEF.md` |

**Security:** Never commit API tokens. Revoke any token pasted in chat.

---

## Glossary (minimal)

| Term | Meaning |
|------|---------|
| **Centroid / contrast vector** | Average activation difference between two behaviors (e.g. panic vs calm) |
| **Conditional steering** | Apply that vector only when a gate says the current state matches |
| **Demo oracle** | Tiny `demo()` with a known expected return per task |
| **Gate** | Script-written JSON checklist (pass/fail + sabotage probes) |
| **Aider** | Open-source agent that edits files through an LLM tool loop |

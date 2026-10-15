# Centroid Steering Experiment

**What this is:** Portfolio work demonstrating **activation steering** for a coding LLM—vectors, gated injection, live agent runs, honest measurement. **Skills demonstration**, not a product or peer-reviewed benchmark.

| | |
|---|---|
| **GitHub** | [Centroid-Steering-Experiment](https://github.com/quantomania-rudolph/Centroid-Steering-Experiment) |
| **Reviewers** | [FOR_REVIEWERS.md](FOR_REVIEWERS.md) — verify every bullet below in ~5 minutes |
| **Live metrics** | [resume_metrics_snapshot.json](resume_metrics_snapshot.json) ← `python research/compute_resume_stats.py` |
| **Repo home** | [README.md](../README.md) |
| **Last updated** | 2026-10-14 |

**Model:** Qwen2.5-Coder-7B-Instruct (Intel Arc XPU) · **Agent:** Aider + custom OpenAI-compatible server

---

## Headline result (resume / LinkedIn)

On **15** matched tasks with **verified live steering** in logs:

| | With conditional steering | Same tasks, no steering |
|---|--:|--:|
| `demo()` checks pass | **12 / 15 (80.0%)** | **5 / 15 (33.3%)** |
| Lift | **+46.7 pp** | — |
| Relative success | **2.4×** | baseline |

**One line:**

> Built conditional activation steering for a local coding agent; on a 15-task slice with verified inject, demo success rose from **33% to 80%** vs non-steered twins.

**Interview honesty:** Subset with verified inject—not full corpus; not randomized; demos on final on-disk code after full Aider sessions. Per-pair breakdown: snapshot `pair_details`.

---

## Resume bullets (copy-paste)

Evidence column: see **[FOR_REVIEWERS.md](FOR_REVIEWERS.md)** (same numbering).

1. Built an **end-to-end activation steering** pipeline for Qwen2.5-Coder-7B on Intel Arc: contrast vectors from **39** prefix-aligned agent traces, live inject at layer 16, full session logging.

2. Implemented **conditional steering** so injection runs when stream monitors match trained “stub + distress” patterns—not on clean control runs (automated gate proofs in `artifacts/gates/`).

3. Calibrated injection strength (**α₀ = 0.85**) with step-down and holdout checks; fail-closed preflight before long agent jobs.

4. Ran **live Aider experiments** over **129** instrumented sessions; on a **15-task** matched slice with verified inject, **80%** demo pass vs **33%** without steering.

5. Validated on **two unseen** coding tasks via `research/prove_cast_oneoff_live.py` (gates pass, demos pass).

6. Added **unit tests** and gate runners for corpus policy, CAST preflight, and agent stall watchdogs on long XPU generations.

---

## Supporting numbers

| Metric | Value | Source |
|--------|------:|--------|
| Inv1 avg completion tokens (steered vs neutral) | 116 vs 330 (**64.9%** reduction) | `compute_resume_stats.py` |
| Corpus sessions | 129 (56 panic / 60 neutral OK) | `scan_report.json` |
| Natural contrast pairs | 39 @ α₀ **0.85** | `metadata.json` |

**Do not claim:** 100% corpus success, faster inference, production copilot (`production_ready: false` in `caa_steering_status.json`).

---

## Sound bite (~30 seconds)

“I built a local coding-agent setup that **steers** the model mid-generation: a direction from calm vs panic traces, applied only when monitors match the failure mode we trained on. I measured it with demo tests and gate JSON—not as a paper, but to show steering from idea to reproducible runs. On one honest 15-task slice, steered runs beat neutral twins **80% to 33%**.”

---

## Glossary

| Term | Meaning |
|------|---------|
| **Contrast / centroid vector** | Activation difference between two behaviors (panic vs calm) |
| **Conditional steering** | Apply vector only when a gate matches |
| **Demo oracle** | `demo()` with known expected return per task |
| **Gate** | Automated pass/fail JSON (+ sabotage probes) |

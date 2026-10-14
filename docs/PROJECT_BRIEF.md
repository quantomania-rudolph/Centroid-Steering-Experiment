# Centroid Steering Experiment — Project Brief

**Status:** Portfolio / research harness (live conditional steering demonstrated; not production-deployed)  
**Last updated:** 2026-10-14  
**Public name:** Centroid Steering Experiment  
**Repo folder / legacy codename:** `J-Do It` · Python package: `jlens-qwen` (J-Lens on Qwen2.5-Coder-7B-Instruct)

| Doc | Purpose |
|-----|---------|
| [FOR_REVIEWERS.md](FOR_REVIEWERS.md) | **Start here** — resume claims → files & commands |
| [CENTROID_STEERING_EXPERIMENT_RESUME.md](CENTROID_STEERING_EXPERIMENT_RESUME.md) | Recruiter packet & bullets |
| [resume_metrics_snapshot.json](resume_metrics_snapshot.json) | Frozen headline metrics — `python research/compute_resume_stats.py` |
| [README.md](../README.md) | GitHub landing (mirrors headline + claim table) |

---

## One sentence

Demonstrate **skills in activation steering** for agentic code generation: detect in-flight “panic” during decode, apply a calibrated contrast vector **only when gated**, and prove behavior with logs, demos, and automated gates.

---

## Main objective

Build **conditional steering (CAST-style)** on top of **contrast vectors (CAA-style)** for the UNI_CAA01 “in-flight panic” behavior at layer 16: a condition direction gates injection; a behavior direction nudges toward complete, calm implementation when stream monitors fire (stub + distress recipes).

Stack:

- **jlens-qwen / J-Lens** — sparse activation readouts and Jacobian tooling (supporting interpretability work)
- **Chaos monitor bus** — per-token flags during decode on Intel Arc XPU
- **Agent calibration** — Aider × Qwen for matched-prefix panic/calm pairs
- **Gate harnesses** — JSON pass/fail artifacts under `artifacts/gates/`

Longer arc (not required to claim for portfolio): SAE-refined directions and richer J-space reconstruction. Current vectors are mean-difference contrasts—a deliberate baseline.

---

## What is working today

| Component | State |
|-----------|-------|
| fp16 inference on Intel Arc XPU | Shipped |
| Per-token chaos monitor + recipe matcher | Shipped |
| UNI_CAA01 offline vectors + metadata (39 natural pairs, α₀=0.85) | Complete |
| CAST live stream + inject on panic paths | Demonstrated (gate proofs) |
| Corpus + one-off Aider steering runs | Demonstrated |
| Automated gates + sabotage probes | Shipped |
| Production “copilot” / cloud deploy | **Out of scope** |

---

## Strengths (why this is a credible steering demo)

1. **End-to-end** — Vectors, server hook, agent loop, measurement, and gates in one repo.  
2. **Conditional, not always-on** — Steering tied to monitor recipes; clean controls stay quiet in gate harnesses.  
3. **Evidence in artifacts** — Session JSON, gate JSON, recompute script (`research/compute_resume_stats.py`).  
4. **Generalization check** — `prove_cast_oneoff_live.py` on tasks outside the corpus.  
5. **Honest scope** — Subset metrics and limitations documented in the resume packet.

---

## Limitations

1. **Portfolio scope** — Local hardware, one model family, engineering proof not a randomized benchmark.  
2. **Subset metrics** — Headline 80% vs 33% is on 15 inject-verified pairs, not full corpus.  
3. **Cost** — Instrumentation adds per-token work; wall time goes up vs bare inference.  
4. **Multi-turn agents** — Final file may differ from first model turn; demos score on-disk result.  
5. **SAE / formal methods** — Future or adjacent work; not part of shipped claims.

---

## Research anchors

| Work | Role |
|------|------|
| Rimsky et al. 2024 (CAA) | Mean activation difference steering |
| Lee et al. 2024 (conditional steering) | Gate on condition, not behavior alone |
| Jacobian Lens | Sparse readouts in this repo’s J-Lens path |

---

## Roadmap (optional, post-portfolio)

1. Broader corpus + harder stress tasks  
2. SAE or richer features for cleaner steering axes  
3. Multi-vector / multi-layer orchestration  

---

## Key artifacts

| Path | Contents |
|------|----------|
| `docs/CENTROID_STEERING_EXPERIMENT_RESUME.md` | Portfolio packet + headline metrics |
| `artifacts/caa/UNI_CAA01/` | Vectors, metadata, replay assets |
| `artifacts/gates/` | Gate pass/fail JSON |
| `research/prove_cast_oneoff_live.py` | OOD live proof harness |
| `research/compute_resume_stats.py` | Headline demo stats |
| `research/CAA_FINDINGS_SUMMARY.md` | Study notes |

---

## Committee / interview framing

> This repo exists to show I can **design, implement, and measure activation steering** in a real agent loop—not to claim a finished product. The chaos layer answers *when* the model looks stuck; conditional injection answers *how* to nudge; gates and demos answer *whether* we can trust a run. Deeper interpretability (J-Lens, SAE) sits alongside as supporting tooling.

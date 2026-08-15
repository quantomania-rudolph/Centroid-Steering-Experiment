# J-Lens × CAST Steering — Project Brief

**Status:** In progress  
**Last updated:** 2026-08-14  
**Repository:** `J-Do It` (J-Lens on Qwen2.5-Coder-7B-Instruct)  
**Part of:** A broader portfolio of interpretability and agent-stability research projects

---

## One sentence

Build a live monitoring and conditional steering pipeline that detects in-flight “panic” during agentic code generation and applies calibrated activation steering only when the model is actually distressed—not when it is merely uncertain or already self-correcting.

---

## Main objective

Develop **Conditional Activation Steering (CAST)** for agentic LLM coding loops: a dual-vector system where a **condition vector** gates injection and a **behavior vector** steers toward calm, complete implementation. The immediate target is **UNI_CAA01 — In-Flight Panic De-escalation** (layer 16 on Qwen2.5-Coder-7B), triggered when stream-time chaos signals (`B_stub`, `B_distress`, `B_STUB_DISTRESS`) indicate the model is apologizing, suggesting `pip install`, or emitting stubs mid-generation.

This project sits on a hybrid stack:

- **J-Lens** (Jacobian lens) for sparse activation-space readouts and reconstruction
- **Chaos monitor bus** (A–G atomic alarms → recipe IDs) for ultra-low-latency per-token flagging during decode on Intel Arc XPU
- **Agentic calibration loops** (Aider × Qwen) to collect matched-prefix pos/neg contrast pairs for offline vector extraction

**Strategic line of work:** This is an intermediate step toward **more accurate steering vectors using Sparse Autoencoders (SAE)** and **improved activation reconstruction** in J-space. Current vectors use mean-difference contrasts (Rimsky et al. CAA); the longer arc is SAE-derived features for cleaner axes and better reconstruction fidelity before live injection.

---

## What is working today

| Component | State |
|-----------|-------|
| fp16 inference on Intel Arc XPU | Shipped |
| Per-token chaos monitor (A–G, recipe matcher, SSE stream) | Shipped |
| Empirical CAA study (129+ agentic invocations cross-referenced) | Complete |
| UNI_CAA01 offline calibration artifacts (`û_cond`, `û_beh`, thresholds) | Complete |
| CAST shadow mode (evaluate gate strength without injecting) | In progress |
| CAST live injection | Not yet shipped |
| Gate proofs (`cast_stream_live`, sabotage checks) | In progress |

---

## Strengths

1. **Evidence-grounded design** — Steering recommendations come from controlled Aider rounds and historical stress runs, not intuition. Signals that self-resolve (`F_LENS`, `A_UNCERTAIN`, lone `B_distress`) are explicitly excluded from CAA gating.

2. **CAST over unconditional CAA** — Following Lee et al. (2024), injection is gated on condition similarity, not behavior similarity. Sabotage tests (shuffled `û_cond`, swapped thresholds, stream dynamics parity) validate that the gate discriminates real panic from noise.

3. **Stream-time vs post-hoc distinction** — The monitor fires during generation (10–200 tokens before turn end), enabling mid-stream rescue rather than only retry-on-next-turn. Post-hoc stub grep is reserved for offline validation and next-invocation gating.

4. **Low-latency hot path** — Flag computation reuses already-paid tensors; recipe evaluation is a fixed bitmask lookup, not ML on the token loop. Designed for Arc-class hardware budgets.

5. **Reproducible gate infrastructure** — Automated gate scripts write JSON artifacts (`artifacts/gates/`) with pass/fail checks, sabotage probes, and environment metadata for committee and audit review.

6. **Agentic systems orchestration** — End-to-end workflow (server, monitor, calibration miners, gate proofs, dashboard) was built through **fully agentic orchestration**: AI agents implemented modules, ran experiments, and drafted specs; the human researcher made critical design decisions, reviewed evidence, and brainstormed with agents on CAST gating, vector semantics, and failure modes.

---

## Limitations

1. **Injection not production-ready** — Shadow evaluation is live; actual residual-stream injection is the current milestone. “Monitor shipped; injection next” is the honest state.

2. **Model and hardware scope** — Qwen2.5-Coder-7B-Instruct on Intel Arc XPU. Generalization to other models, GPUs, or cloud H100 runs is planned but not validated here.

3. **Vector methodology** — Current `û_cond` / `û_beh` use Method B matched-prefix mean-difference contrasts. These are a baseline, not the target SAE-refined directions. Reconstruction quality in sparse J-space is gated (rel_mse targets) but SAE-based steering is future work.

4. **Calibration data gaps** — Some study rounds (combo scheduler, trajectory tasks) were blocked by server instability. Harder tasks with truly unresolved stub artifacts need additional stress rounds.

5. **Clean-run dominance in outcomes** — Many analyzed invocations ended `RESOLVED` with `stub_score=0`; linking monitor fires to *failed* outcomes requires harder corpora and longer agentic sessions.

6. **Single primary vector family** — Eight CAA candidates were ranked; production focus is UNI_CAA01. Multi-vector orchestration and cross-layer injection (L10/L16/L22) remain design space.

---

## Research anchors

| Work | Role in this project |
|------|---------------------|
| Rimsky et al. 2024 (CAA) | Mean activation difference steering |
| Lee et al. 2024 (CAST) | Conditional gating beats unconditional injection |
| Zou et al. 2023 (RepE) | Behavioral opposites, not topic pairs |
| Jacobian Lens | Sparse J-space readouts and reconstruction |

---

## Roadmap (high level)

1. **Now** — Complete CAST shadow → injection path; pass `cast_stream_live` gate with live SSE parity
2. **Next** — Expand calibration corpus (STEM25, expr_compiler stress, combo tasks)
3. **Then** — SAE feature discovery on panic/calm activation manifolds; replace or refine mean-difference vectors
4. **Then** — Improve J-space reconstruction accuracy as a prerequisite for trustworthy sparse steering
5. **Later** — Multi-vector CAST orchestration across agentic tool loops beyond Aider

---

## How to describe this to a committee

> This is one project in a larger interpretability and agent-stability portfolio. We are building the **sensing and gating layer** that must exist before any steering vector—CAA, SAE-derived, or otherwise—can be applied safely during live agentic coding. The chaos monitor tells us *when* the model is panicking; CAST tells us *how strongly* to steer; UNI_CAA01 is the first behavior vector. The work was executed through agentic systems orchestration with human oversight on architecture, evidence review, and safety gates. Status: **in progress**—monitor and offline calibration complete; live conditional injection is the active milestone toward SAE-accurate vectors and better reconstruction.

---

## Key artifacts for reviewers

| Path | Contents |
|------|----------|
| `research/CAA_FINDINGS_SUMMARY.md` | Executive brief from error-combination study |
| `research/notes/true_UNICAA01_cast.md` | UNI_CAA01 CAST production spec |
| `docs/LIVE_MONITOR_CAA_PLAN.md` | Monitor bus architecture and latency budgets |
| `research/evidence and CAA_vectors.md` | Evidence base and vector production spec |
| `artifacts/caa/UNI_CAA01/` | Calibrated condition/behavior vectors and replay JSONL |
| `artifacts/gates/` | Automated gate pass/fail JSON |
| `research/prove_cast_stream_live.py` | Live CAST shadow stream proof (Gate A) |

# Centroid Steering Experiment

[![reviewer-claims](https://github.com/quantomania-rudolph/centroid-steering-experiment/actions/workflows/reviewer-claims.yml/badge.svg)](https://github.com/quantomania-rudolph/centroid-steering-experiment/actions/workflows/reviewer-claims.yml)

**Portfolio demonstration** of **activation steering** for a local coding LLM (Qwen2.5-Coder-7B + Aider on Intel Arc). Shows end-to-end skills: contrast vectors, **gated** mid-generation injection, agent instrumentation, and measured outcomes—not a shipped copilot or formal benchmark.

**GitHub About:** Portfolio — conditional activation steering for a coding agent. Reviewers → [docs/FOR_REVIEWERS.md](docs/FOR_REVIEWERS.md).

| | |
|---|---|
| **Reviewers** | **[docs/FOR_REVIEWERS.md](docs/FOR_REVIEWERS.md)** — claim → evidence map (5‑min check, no GPU) |
| **Resume packet** | [docs/CENTROID_STEERING_EXPERIMENT_RESUME.md](docs/CENTROID_STEERING_EXPERIMENT_RESUME.md) |
| **Metrics (JSON)** | [docs/resume_metrics_snapshot.json](docs/resume_metrics_snapshot.json) |
| **Doc index** | [docs/README.md](docs/README.md) |
| **Model** | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) · **α₀ = 0.85** · **39** CAA pairs |

---

## Headline result (same as resume)

On **15** corpus task pairs with **verified live steering** in session logs (`cast_enabled`, `cast_alpha_0 = 0.85`, ≥1 inject token):

| | Steered (panic + CAST) | Neutral twin |
|---|--:|--:|
| `demo()` oracle pass | **12 / 15 (80.0%)** | **5 / 15 (33.3%)** |
| Lift | **+46.7 pp** | — |
| Relative success | **2.4×** | baseline |
| Avg inv1 completion tokens | **116** | **330** (**64.9%** fewer on this slice) |

**Corpus scale:** **129** session files (**56** panic OK / **60** neutral OK) — [scan_report.json](artifacts/caa/UNI_CAA01_corpus/scan_report.json).

**Verify without GPU** (rewrites snapshot + prints counts):

```powershell
python research/compute_resume_stats.py
```

**Caveats (read before citing):** Subset with verified inject, not all ~66 pairs; not randomized A/B; demos score **final** on-disk `main.py` after full Aider sessions. `production_ready` is **false** in [caa_steering_status.json](artifacts/gates/caa_steering_status.json).

---

## Resume bullets → where to look

Full table: **[docs/FOR_REVIEWERS.md](docs/FOR_REVIEWERS.md)**. Short version:

| # | Claim | Primary evidence |
|---|--------|------------------|
| 1 | End-to-end steering; **39** pairs; L16 inject; logging | [metadata.json](artifacts/caa/UNI_CAA01/metadata.json), [serve_qwen_openai.py](experiments/serve_qwen_openai.py) |
| 2 | **Conditional** inject (panic yes, clean controls quiet) | [cast_stream_live_gate.json](artifacts/gates/cast_stream_live_gate.json), [cast_inject_live_gate.json](artifacts/gates/cast_inject_live_gate.json) |
| 3 | **α₀ = 0.85** calibration + preflight | [caa_alpha0_4c_gate.json](artifacts/gates/caa_alpha0_4c_gate.json), [cast_run_preflight.py](src/jlens_qwen/caa/cast_run_preflight.py) |
| 4 | **129** sessions; **80% vs 33%** on **15** pairs | [scan_report.json](artifacts/caa/UNI_CAA01_corpus/scan_report.json), `compute_resume_stats.py` |
| 5 | **2** OOD tasks (min-heap, min-stack) | [cast_oneoff_panic_live_proof.json](artifacts/gates/cast_oneoff_panic_live_proof.json), [cast_oneoff_b02_live_proof.json](artifacts/gates/cast_oneoff_b02_live_proof.json) |
| 6 | Tests: corpus policy, preflight, watchdog | `pytest tests/unit/test_unicaa01_corpus_policy.py tests/unit/test_cast_run_preflight.py tests/unit/test_aider_watchdog.py -q` |

**Primary gates (all `pass: true` in repo):** stream live · inject live · alpha0 4c · one-off panic · one-off b02 (see `gates_pass` in snapshot).

---

## How it works

```text
Broken Python stub → Aider fixes one file
        ↓
Token stream: monitors flag stub-like / stressed patterns
        ↓
Gate open → add precomputed contrast vector (layer 16)
        ↓
Session JSON + demo tests + gate artifacts
```

**Code map**

| Path | Role |
|------|------|
| `experiments/serve_qwen_openai.py` | Inference + monitors + steering hook |
| `experiments/run_aider_agent.py` | Aider wrapper, sessions, watchdogs |
| `src/jlens_qwen/caa/` | Conditional steering math and gates |
| `research/run_unicaa01_corpus.py` | Panic / neutral corpus runs |
| `research/prove_cast_oneoff_live.py` | Live proof on unseen tasks |
| `research/compute_resume_stats.py` | Headline stats + snapshot JSON |

Python package name: **`jlens-qwen`** (Jacobian lens tooling; legacy *J-Lens*). GitHub repo slug may still be **`J-Do-It`** until renamed in Settings.

---

## Setup

```powershell
git clone https://github.com/quantomania-rudolph/centroid-steering-experiment.git
cd centroid-steering-experiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
pip install -e .
pip install "git+https://github.com/anthropics/jacobian-lens.git"
```

Download weights and smoke XPU before long steering runs ([experiments/README.md](experiments/README.md)).

---

## Optional: re-run live steering proof

Needs model + server on XPU:

```powershell
python research/prove_cast_oneoff_live.py --task panic
```

---

## J-Lens phase gates (hardware / Jacobian fit)

```powershell
python -m tests.gates.run_all --through 0
python scripts\probe_hardware.py
python scripts\download_model.py
python scripts\smoke_infer_xpu.py
python -m tests.gates.run_all --through 6
```

Jacobian smoke / production fit: see [PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) or previous README sections in git history; optional for steering claims.

---

## Secrets

Never commit API keys or personal access tokens. Revoke any token that was shared in chat.

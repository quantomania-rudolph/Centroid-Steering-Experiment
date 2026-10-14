# Centroid Steering Experiment

**Portfolio project:** end-to-end **activation steering** for a local coding LLM—contrast vectors, gated mid-generation injection, Aider agent runs, and measured outcomes. This demonstrates engineering and research skills in steering; it is **not** a production copilot or a formal benchmark paper.

| | |
|---|---|
| **Model** | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| **Hardware** | Intel Arc XPU (fp16), Windows-friendly local stack |
| **Agent** | [Aider](https://aider.chat/) + custom OpenAI-compatible server (`experiments/serve_qwen_openai.py`) |
| **Python package** | `jlens-qwen` (Jacobian lens tooling; legacy name *J-Lens*) |

**More detail for recruiters / reviewers:** [docs/CENTROID_STEERING_EXPERIMENT_RESUME.md](docs/CENTROID_STEERING_EXPERIMENT_RESUME.md) · [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md)

**Metrics snapshot (auto-generated):** [docs/resume_metrics_snapshot.json](docs/resume_metrics_snapshot.json) · last verified **2026-10-14** (UTC)

---

## Result (honest slice)

On **15 matched tasks** where steering fired in session logs (`cast_enabled`, α₀=0.85, `inject_count ≥ 1` in corpus `_latest.json`):

| | Conditional steering | No steering (twin runs) |
|---|--:|--:|
| `demo()` oracle pass | **12 / 15 (80.0%)** | **5 / 15 (33.3%)** |
| Lift vs neutral | **+46.7 pp** | — |
| Relative success | **~2.4×** | baseline |
| Avg inv1 completion tokens | **116** | **330** (~**65%** fewer on this slice) |

| Corpus (scan report) | Count |
|----------------------|------:|
| Session files | 129 |
| Panic runs OK | 56 |
| Neutral runs OK | 60 |
| CAA natural pairs (`metadata.json`) | 39 @ **α₀ = 0.85** |

**Gate artifacts (all `pass: true` in repo):** `cast_stream_live_gate.json`, `cast_inject_live_gate.json`, `caa_alpha0_4c_gate.json`, `cast_oneoff_panic_live_proof.json`, `cast_oneoff_b02_live_proof.json`.

Refresh numbers (no GPU required; rewrites the snapshot JSON):

```powershell
python research/compute_resume_stats.py
```

**Caveat:** Subset with verified inject—not the full 66-pair corpus; not randomized A/B. Demos score final on-disk `main.py` after full Aider sessions. See the resume doc for interview-safe wording.

---

## What this repo shows

- **Contrast steering vectors** from prefix-aligned calm vs “panic” agent traces (`artifacts/caa/UNI_CAA01/`, 39 natural pairs in metadata).
- **Conditional injection**—add a layer-16 direction only when stream monitors match a trained recipe (e.g. stub + distress), not on every token.
- **Live instrumentation**—per-token logs, inject metadata, automated **gate** JSON under `artifacts/gates/`.
- **Generalization check**—one-off tasks outside the corpus (`research/prove_cast_oneoff_live.py`).

```text
Broken Python stub → Aider fixes one file
        ↓
Token stream: monitors score stub-like / stressed patterns
        ↓
Gate open → add precomputed steering vector at decode time
        ↓
Session JSON + demo tests + gate artifacts
```

---

## Repository map (high level)

| Path | Purpose |
|------|---------|
| `experiments/serve_qwen_openai.py` | Inference server (monitors + steering hook) |
| `experiments/run_aider_agent.py` | Aider wrapper, sessions, watchdogs |
| `src/jlens_qwen/caa/` | CAST / stream gate / inject math |
| `research/run_unicaa01_corpus.py` | Corpus panic/neutral pairs |
| `research/prove_cast_oneoff_live.py` | Live proof on unseen tasks |
| `research/compute_resume_stats.py` | Headline demo stats |
| `tests/` | Unit tests + phase gates |

---

## Setup

```powershell
git clone https://github.com/quantomania-rudolph/J-Do-It.git
cd J-Do-It
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
pip install -e .
pip install "git+https://github.com/anthropics/jacobian-lens.git"
```

Download weights and smoke-test XPU (see phase gates below) before long steering runs.

---

## Run a steering proof

Requires model on disk and the Qwen server running (see `experiments/README.md`).

```powershell
python research/prove_cast_oneoff_live.py --task panic
# optional: python research/prove_cast_oneoff_live.py --task b02
```

Pass/fail JSON: `artifacts/gates/cast_oneoff_*_live_proof.json`.

---

## Phase gates (hardware + J-Lens pipeline)

```powershell
python -m tests.gates.run_all --through 0
python scripts\probe_hardware.py
python scripts\download_model.py
python scripts\smoke_infer_xpu.py
python -m tests.gates.run_all --through 6
```

### Jacobian fit (optional, CPU-heavy)

**Smoke** — 1 layer, 1 prompt:

```powershell
python scripts\fit_jacobian.py --n-prompts 1 --source-layers 15-15 --device cpu --dim-batch 8 --max-seq-len 64 --out artifacts\lenses\qwen_jacobian_lens_L15.pt
python scripts\verify_lens_smoke.py
```

**Production** (only after smoke PASS) — layers 0–26, 100 prompts; many hours on CPU:

```powershell
python scripts\run_production_fit.py
```

Or: `python scripts\wait_and_verify_smoke.py --launch-production`

### Slice viewer

```powershell
python scripts\slice_view.py
```

Open `artifacts\slices\slice.html`.

---

## Naming note

- **Project title:** Centroid Steering Experiment (README + docs).
- **GitHub repo slug** may still be `J-Do-It` until you rename it under **Settings → General** (e.g. `centroid-steering-experiment`); update `git remote` after rename.
- **Local folder** may still be `J-Do It` on disk—paths in old artifacts can reference it.

---

## License / secrets

Do not commit API keys or tokens. Large artifacts and run outputs may be local-only—check `.gitignore` before pushing.

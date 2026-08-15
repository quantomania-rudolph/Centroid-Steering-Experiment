# Critical audit — H100 rental vs local production

Verdict: **YES — same production artifacts, same J-space decoding capability**, if we keep the config locks below. The rental path fits the lens; local code does readouts/verify/decode. Auto-destroy is supported so billing stops even if you fall asleep.

## What "production-level" means here

| Artifact / capability | Local CPU path | H100 rental path | Same? |
|-----------------------|----------------|------------------|-------|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` | same (`fit_config.yaml`) | Yes |
| Source layers | 8–26 | 8–26 | Yes |
| Target | L27 | L27 | Yes |
| Prompts | `corpus_100.json` × 100 | same file bundled | Yes |
| Estimator | `jlens.fit` / `jacobian_for_prompt` | `jlens.fit` from anthropics/jacobian-lens | Yes (same library) |
| `max_seq_len` / `skip_first` | 128 / 16 | 128 / 16 | Yes |
| `dim_batch` | 14 (CPU RAM) | 64 (H100 VRAM) | **Math-equivalent** — partitions of d_model only; total FLOPs unchanged |
| Output `.pt` format | `FittedJacobianLens` (`J`, `n_prompts`, `d_model`) | `fit_on_gpu.py` writes same keys | Yes |
| L8–L26 readouts | `run_all_layer_readouts.py` | same, via `integrate_local.py` | Yes |
| Apply / transport / unembed | local `jlens_qwen` | local after download | Yes |
| Workspace / slice / gates | local | local on rented lens | Yes |

## What you will have in the codebase after a good run

1. `artifacts/lenses/qwen_jacobian_lens.pt` — 19 Jacobians for decode/steer
2. `artifacts/readouts/production/L{8..26}_readout.txt` (+ `.json`)
3. `artifacts/gates/rental_fit_meta.json` + production verify JSON
4. Everything downstream (apply, workspace_detect, slice_view) works on that lens

## Gaps that were real (and fixes)

| Gap | Risk | Fix |
|-----|------|-----|
| No auto-destroy | Billing if you sleep | `destroy_vast_instance.py` + `run_full_pipeline.py` `finally` |
| Smoke verify only checked L15 | Weak for full lens | `verify_production_lens.py` checks layers 8–26 present + finite |
| Docker was optional/unused | Confusion | Primary path is Vast PyTorch template + `run_fit.sh` (Docker optional) |
| `VAST_API_KEY` unused | No remote destroy | Required for auto-destroy; `VAST_INSTANCE_ID` after launch |
| No hard time cap | Runaway bill | `MAX_BILLING_MINUTES` + watchdog destroy |
| HF public lens is *non*-Coder 7B | Wrong model if copied | We fit **Coder** ourselves — do not download the public Instruct lens |

## Equivalence notes (honest)

- **fp16 CPU vs bf16 GPU weights**: Jacobian accumulators are float32; saved lens is fp16. Expect tiny numeric noise, not structural difference. Same as any GPU vs CPU fit of this estimator.
- **`dim_batch`**: does not change the estimator; only memory/time tradeoff.
- **Checkpoint format mid-fit** differs from final lens; we only download the final `qwen_jacobian_lens.pt` written by `save_fitted_lens`.
- **Solid when**: production verify passes (19 layers, shapes, finite, apply smoke) + readouts all_ok.

## What is NOT solid without you

- Launching / paying Vast (you)
- Pasting `HF_TOKEN`, `VAST_API_KEY`, SSH fields, `VAST_INSTANCE_ID` into `secrets/secrets.env` (you)
- Confirming destroy returned success (pipeline does it; glance at console once)

## Docker vs Vast

- **Recommended**: Vast marketplace PyTorch image + SSH + `run_fit.sh` (no custom image build).
- **Dockerfile**: optional convenience; not required for correctness.
- Correctness comes from `fit_on_gpu.py` + `fit_config.yaml` + your `corpus_100.json`, not from Docker.
